// Author: APD team, except where source was noted
#define _POSIX_C_SOURCE 200112L /* Or higher */

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <stdbool.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// Data structure that contains all the information we send to the functions that are paralyzed
struct Arguments {
    int thread_id;
    int thread_number;
    int step_x;
    int step_y;
    int p;
    int q;
    bool need_rescale;
    ppm_image *image;
    ppm_image *old_image;
    ppm_image **contour_image;
    pthread_barrier_t *barrier;
    unsigned char **grid;
};

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
void sample_grid(int thread_id, int thread_number, int p, int q, ppm_image *image, int step_x,
                int step_y, unsigned char **grid, unsigned char sigma) {

    // calculating the points where each thread will start and end based on the id of the thread
    int start = thread_id * (double) p / thread_number;
    int end = fmin(p, (thread_id + 1) * (double) p / thread_number);

    // paralyzing the first loop so that each thread modifies a set of lines from the grid matrix
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }

    // calculating the points where each thread will start and end based on the id of the thread
    start = thread_id * (double) p / thread_number;
    end = fmin(p, (thread_id + 1) * (double) p / thread_number);

    // paralyzing the next two loops so each thread will have it's own rows/columns to change
    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }

    // calculating the points where each thread will start and end based on the id of the thread
    start = thread_id * (double) q / thread_number;
    end = fmin(q, (thread_id + 1) * (double) q / thread_number);

    for (int j = start; j < end; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(int thread_id, int thread_number, int p, int q, ppm_image *image, unsigned char **grid,
            ppm_image **contour_image, int step_x, int step_y) {

    // calculating the points where each thread will start and end based on the id of the thread
    int start = thread_id * (double) p / thread_number;
    int end = fmin(p, (thread_id + 1) * (double) p / thread_number);

    // paralyzing the action of marching the image by having each thread do a certain part of it, divided by lines
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_image[k], i * step_x, j * step_y);
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

void rescale_image(int thread_id, int thread_number, ppm_image *image, ppm_image *old_image) {
    uint8_t sample[3];

    // calculating the points where each thread will start and end based on the id of the thread
    int start = thread_id * (double) image->x / thread_number;
    int end = fmin(image->x, (thread_id + 1) * (double) image->x / thread_number);

    // paralyzing by lines
    // use bicubic interpolation for scaling
    for (int i = start; i < end; i++) {
        for (int j = 0; j < image->y; j++) {
            float u = (float)i / (float)(image->x - 1);
            float v = (float)j / (float)(image->y - 1);
            sample_bicubic(old_image, u, v, sample);

            image->data[i * image->y + j].red = sample[0];
            image->data[i * image->y + j].green = sample[1];
            image->data[i * image->y + j].blue = sample[2];
        }
    }
}

// function that contains the programs which need to be paralyzed
void *solve(void *arg) {
    // extracting the data we need
    struct Arguments *arguments = (struct Arguments *)arg;

    bool need_rescale = arguments->need_rescale;

    int thread_id = arguments->thread_id;
    int thread_number = arguments->thread_number;

    int p = arguments->p;
    int q = arguments->q;

    int step_x = arguments->step_x;
    int step_y = arguments->step_y;

    unsigned char **grid = arguments->grid;

    ppm_image **contour_image = arguments->contour_image;

    ppm_image *image = arguments->image;

    ppm_image *old_image = arguments->old_image;

    pthread_barrier_t *barrier = arguments->barrier;

    // we rescale the image only if needed
    if (need_rescale) {
        // 1. Rescale the image
        rescale_image(thread_id, thread_number, image, old_image);
    }

    // waiting for all the threads to finish the rescaling process
    pthread_barrier_wait(barrier);

    // 2. Sample the grid
    sample_grid(thread_id, thread_number, p, q, image, step_x, step_y, grid, SIGMA);

    // waiting for all the threads to finish so we can have the grid matrix complete
    pthread_barrier_wait(barrier);

    // 3. March the squares
    march(thread_id, thread_number, p, q, image, grid, contour_image, step_x, step_y);

    // terminating the calling thread
    pthread_exit(NULL);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file>\n");
        return 1;
    }

    // getting the number of threads we will be using and declaring the array of the threads ids
    int thread_number = atoi(argv[3]);
    pthread_t tid[thread_number];

    ppm_image *image = read_ppm(argv[1]);

    // creating a copy of the initial image that we will be using for the rescale process
    ppm_image *old_image = image;

    // defining and initializing our barrier with the number of threads we're supposed to wait on
    pthread_barrier_t barrier;

    pthread_barrier_init(&barrier, NULL, thread_number);

    // 0. Initialize contour map
    ppm_image **contour_map = init_contour_map();

    // variable that tells us when to rescale an image
    bool need_rescale = false;

    // we only rescale downwards
    if (image->x > RESCALE_X || image->y > RESCALE_Y) {
        // alloc memory for image
        ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));

        if (!new_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }

        new_image->x = RESCALE_X;
        new_image->y = RESCALE_Y;

        new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));

        if (!new_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }

        // replacing our image with the new one
        image = new_image;

        need_rescale = true;
    }

    // calculating the dimensions of the grid matrix
    int p = image->x / STEP;
    int q = image->y / STEP;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    // creating for each thread the data structure with info that the functions will need
    for (int i = 0; i < thread_number; i++) {
        struct Arguments *arguments = malloc(sizeof(struct Arguments));

        arguments->thread_id = i;
        arguments->thread_number = thread_number;
        arguments->step_x = STEP;
        arguments->step_y = STEP;
        arguments->need_rescale = need_rescale;
        arguments->image = image;
        arguments->contour_image = contour_map;
        arguments->barrier = &barrier;
        arguments->p = p;
        arguments->q = q;

        arguments->old_image = old_image;

        arguments->grid = grid;

        pthread_create(&tid[i], NULL, solve, arguments);
    }

    // waiting for the threads to finish
    for (int i = 0; i < thread_number; i++) {
        pthread_join(tid[i], NULL);
    }

    // destroying the barrier
    pthread_barrier_destroy(&barrier);

    // 4. Write output
    write_ppm(image, argv[2]);

    free_resources(image, contour_map, grid, STEP);

    return 0;
}
