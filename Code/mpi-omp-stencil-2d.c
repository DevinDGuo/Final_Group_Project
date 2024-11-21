#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "utilities.h"
#include "timer.h"
#include <mpi.h>
#include "MyMPI.h"

void printUsage() {
    printf("Usage: mpirun -np <num of processes> ./mpi-stencil-2d <num iterations> <input file> <output file> <debug level> <all-stacked-file-name.raw (optional)>\n");
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    if (argc < 5 || argc > 6) {
        printUsage();
        MPI_Finalize();
        return 1;
    }

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double overall_start, overall_end, work_start, work_end, other_total = 0.0;

    // Start overall timing
    GET_TIME(overall_start);

    int iterations = atoi(argv[1]);
    char *inFile = argv[2];
    char *outFile = argv[3];
    double **matrix1;
    int debug_level = atoi(argv[4]);
    char *allIterationsFile = NULL;

    if (argc == 6) {
        allIterationsFile = argv[5]; // Optional argument for stacked file
    }
    
    double** matrix;
    int rows, cols;

    if (rank == 0) {
        if (debug_level == 1) {
            struct stat inFileStat;

            printf("Running with %d threads. \n", size);
            // Check the input file size
            if (stat(inFile, &inFileStat) == 0) {
                long inFileSize = inFileStat.st_size;
                printf("Reading from Input file: %s with size of %ld bytes \n", inFile, inFileSize);
            } else {
                perror("Error getting input file size");
            }

        }
    }

    read_row_striped_matrix_halo(inFile, (void***)&matrix, MPI_DOUBLE, &rows, &cols, MPI_COMM_WORLD);
    exchange_row_striped_values((void***)&matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (debug_level == 0) {
            if (matrix == NULL) {
                printf("Error: Failed to read matrix from file.\n");
                return 1;
            }
        }
    }

    if (size > rows) {
        fprintf(stderr, "Error: Number of processes (%d) cannot exceed the number of rows (%d).\n", size, rows);
        return 1;
    }


    // Work timing
    GET_TIME(work_start);

    if (rank == 0) {
        if (debug_level == 1) {
            printf("Starting stencil operation...\n");
        }
    }

    read_row_striped_matrix_halo(inFile, (void***)&matrix1, MPI_DOUBLE, &rows, &cols, MPI_COMM_WORLD);

    #pragma omp parallel default(none)
    {
        for (int i = 0; i < iterations; i++) {

            // Apply stencil operation
            stencil2D_MPI_OMP(matrix, matrix1, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
            if (debug_level == 2) {
                MPI_Barrier(MPI_COMM_WORLD);
                print_row_striped_matrix_halo((void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
            }
            // Swap pointers for next iteration
            double **temp = matrix1;
            matrix1 = matrix;
            matrix = temp; 

            exchange_row_striped_values((void***)&matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
        }
    }
    if (debug_level == 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        print_row_striped_matrix_halo((void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (debug_level == 1) {
            printf("Ending stencil operation...\n");
        }
    }

    GET_TIME(work_end);

    if (rank == 0) {
        if (debug_level == 1) {
            printf("Writing data to %s .\n", outFile);
        }
    }

    write_row_striped_matrix_halo(outFile, (void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
    
    my_free((void **)matrix);
    my_free((void **)matrix1);
    // End overall timing
    GET_TIME(overall_end);


    // Calculate times
    double overall_time = overall_end - overall_start;
    double work_time = work_end - work_start - other_total;
    double total_other_time = overall_time - work_time;
    
    if (rank == 0) {
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
    }

    if (rank == 0) {
        if (debug_level == 0 || debug_level == 1 || debug_level == 2) {
            printf("Time Overall: %.5f seconds\n", overall_time);
            printf("Time Computation: %.5f seconds\n", work_time);
            printf("Time Other: %.5f seconds\n", total_other_time);
        }
    }

    MPI_Finalize();

    return 0;
}