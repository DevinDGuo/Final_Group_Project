#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <mpi.h>
#include "utilities.h"
#include "timer.h"

#define dtype double

void printUsage() {
    printf("Usage: mpirun -np <num processes> ./program <num iter.> <infile> <outfile> <debug level> [<all-stacked-file-name.raw>]\n");
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int num_process, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 5 || argc > 6) {
        if (rank == 0) printUsage();
        MPI_Finalize();
        return 1;
    }

    double overall_start, overall_end, work_start, work_end, other_total = 0.0;

    // Start overall timing
    GET_TIME(overall_start);

    int iterations = atoi(argv[1]);
    char *inFile = argv[2];
    char *outFile = argv[3];
    int debug_level = atoi(argv[4]);
    char *allIterationsFile = NULL;
    if (argc == 6) {
        allIterationsFile = argv[5]; // Optional argument for stacked file
    }

    double **matrix = NULL, **matrix1 = NULL;
    int rows, cols;

    // Debug: Input file info
    if (debug_level == 1 && rank == 0) {
        struct stat inFileStat;
        if (stat(inFile, &inFileStat) == 0) {
            printf("Reading from input file: %s (size: %ld bytes)\n", inFile, inFileStat.st_size);
        } else {
            perror("Error getting input file size");
        }
    }

    // Read matrix
    read_row_striped_matrix_halo(inFile, (void***)&matrix, MPI_DOUBLE, &rows, &cols, MPI_COMM_WORLD);
    if (matrix == NULL) {
        if (rank == 0) fprintf(stderr, "Error: Failed to read matrix from file.\n");
        MPI_Finalize();
        return 1;
    }

    // Validate number of threads
    if (num_process > rows) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes (%d) cannot exceed the number of rows (%d).\n", num_process, rows);
        }
        free(matrix);
        MPI_Finalize();
        return 1;
    }

    malloc2D(&matrix1, rows, cols);

    // Preserve boundaries
    for (int i = 0; i < cols; i++) {
        matrix1[0][i] = matrix[0][i];
        matrix1[rows - 1][i] = matrix[rows - 1][i];
    }

    // Work timing
    GET_TIME(work_start);

    if (debug_level == 1 && rank == 0) printf("Starting Stencil.\n");

    // stencil2DPThread(iterations, debug_level, rows, cols, matrix, matrix1, allIterationsFile, &other_total, num_process);

    if (debug_level == 1 && rank == 0) printf("Stencil completed for %d iterations.\n", iterations);

    GET_TIME(work_end);

    // Write output matrix
    write_row_striped_matrix_halo(outFile, (void**)matrix1, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);

    // Free memory
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
        free(matrix1[i]);
    }
    free(matrix);
    free(matrix1);

    // End overall timing
    GET_TIME(overall_end);

    // Calculate times
    double overall_time = overall_end - overall_start;
    double work_time = work_end - work_start - other_total;
    double total_other_time = overall_time - work_time;

    if (debug_level >= 0 && rank == 0) {
        printf("Time Overall: %.5f seconds\n", overall_time);
        printf("Time Computation: %.5f seconds\n", work_time);
        printf("Time Other: %.5f seconds\n", total_other_time);
    }

    // Debug: Output file info
    if (debug_level == 1 && rank == 0) {
        struct stat outFileStat;
        if (stat(outFile, &outFileStat) == 0) {
            printf("Output file: %s (size: %ld bytes)\n", outFile, outFileStat.st_size);
        } else {
            perror("Error getting output file size");
        }

        if (allIterationsFile != NULL) {
            struct stat allIterationsFileStat;
            if (stat(allIterationsFile, &allIterationsFileStat) == 0) {
                printf("All iterations file: %s (size: %ld bytes)\n", allIterationsFile, allIterationsFileStat.st_size);
            }
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
