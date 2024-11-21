#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "utilities.h"
#include "timer.h"
#include <omp.h>


void printUsage() {
    printf("Usage: ./pth-stencil-2d <num iterations> <input file> <output file> <debug level> <num threads> <all-stacked-file-name.raw (optional)>\n");
}

int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 7) {
        printUsage();
        return 1;
    }


    double overall_start, overall_end, work_start, work_end, other_total;

    // Start overall timing
    GET_TIME(overall_start);

    int iterations = atoi(argv[1]);
    char *inFile = argv[2];
    char *outFile = argv[3];
    double **matrix1;
    int debug_level = atoi(argv[4]);
    int num_threads = atoi(argv[5]);
    char *allIterationsFile = NULL;

    if (argc == 7) {
        allIterationsFile = argv[6]; // Optional argument for stacked file
    }
    omp_set_num_threads(num_threads);

    
    int rows, cols;

    if (debug_level == 1) {
        struct stat inFileStat;

        printf("Running with %d threads. \n", num_threads);
        // Check the input file size
        if (stat(inFile, &inFileStat) == 0) {
            long inFileSize = inFileStat.st_size;
            printf("Reading from Input file: %s with size of %ld bytes \n", inFile, inFileSize);
        } else {
            perror("Error getting input file size");
        }

    }

    double **matrix = read2DMatrix(&rows, &cols, inFile);

    if (debug_level == 0) {
        if (matrix == NULL) {
            printf("Error: Failed to read matrix from file.\n");
            return 1;
        }
    }

    if (num_threads > rows) {
        fprintf(stderr, "Error: Number of threads (%d) cannot exceed the number of rows (%d).\n", num_threads, rows);
        return 1;
    }
    
    malloc2D(&matrix1, rows, cols);
        
    if (matrix1 == NULL && debug_level == 0) {
        printf("Error: Memory allocation failed for matrix1.\n");
        return 1; 
    }

    for (int i = 0; i < rows; i++){
        matrix1[i][0] = matrix[i][0];
        matrix1[i][cols-1] = matrix[i][cols-1];
    }

    for (int j = 0; j < cols; j++) {
        matrix1[0][j] = matrix[0][j];           // First row
        matrix1[rows - 1][j] = matrix[rows - 1][j]; // Last row
    }
    
    // Work timing
    GET_TIME(work_start);

    if (debug_level == 1) {
        printf("Starting Stencil.\n");
        stencil2DOMP(iterations, debug_level, rows, cols, matrix, matrix1, allIterationsFile, &other_total);
        printf("Stencil Completed for %d iterations. \n", iterations);
    } else {
        stencil2DOMP(iterations, debug_level, rows, cols, matrix, matrix1, allIterationsFile, &other_total);
    }

    GET_TIME(work_end);

    if (debug_level == 1) {
        printf("Writing data to %s .\n", outFile);
    }
    
    if (write2DMatrixToFile(matrix, rows, cols, outFile) != 0) {
        if (debug_level == 0) {
            printf("Error: Failed to write matrix to file: %s\n", outFile);
        }
        free(matrix);
        free(matrix1);
        return 1;  // Error writing matrix
    }

    free(matrix);
    free(matrix1);

    // End overall timing
    GET_TIME(overall_end);

    // Calculate times
    double overall_time = overall_end - overall_start;
    double work_time = work_end - work_start - other_total;
    double total_other_time = overall_time - work_time;

    if (debug_level == 1) {
        struct stat outFileStat, allIterationsFileStat;

        // Check the output file size
        if (stat(outFile, &outFileStat) == 0) {
            long outFileSize = outFileStat.st_size;
            printf("Output file: %s with size of %ld bytes \n", outFile, outFileSize);
        } else {
            perror("Error getting output file size");
        }

        // Check the all iterations file size
        if (allIterationsFile != NULL && stat(allIterationsFile, &allIterationsFileStat) == 0) {
            long allIterationsFileSize = allIterationsFileStat.st_size;
            printf("All iterations file: %s with size of %ld bytes \n", allIterationsFile, allIterationsFileSize);
        }
    }

    if (debug_level == 0 || debug_level == 1 || debug_level == 2) {
        printf("Time Overall: %.5f seconds\n", overall_time);
        printf("Time Computation: %.5f seconds\n", work_time);
        printf("Time Other: %.5f seconds\n", total_other_time);
    }

    return 0;
}
