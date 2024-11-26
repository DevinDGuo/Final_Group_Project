#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "utilities.h"
#include "timer.h"
#include <mpi.h>
#include <omp.h>
#include "MyMPI.h"

void printUsage() {
    printf("Usage: mpirun -np <num of processes> ./mpi-omp-stencil-2d <num iterations> <input file> <output file> <num omp threads> <debug level> <all-stacked-file-name.raw (optional)>\n");
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Check command-line arguments
    if (argc < 6 || argc > 7) {
        printUsage();
        MPI_Finalize();
        return 1;
    }

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments
    int iterations = atoi(argv[1]);
    char *inFile = argv[2];
    char *outFile = argv[3];
    int num_threads = atoi(argv[4]);
    int debug_level = atoi(argv[5]);
    char *allIterationsFile = (argc == 7) ? argv[6] : NULL;

    double overall_start, overall_end, work_start, work_end, other_total = 0.0;

    double other_start, other_end;

    // Start overall timing
    GET_TIME(overall_start);

    double **matrix, **matrix1;
    int rows, cols;

    if (rank == 0 && debug_level == 1) {
        struct stat inFileStat;
        printf("Running with %d MPI processes and %d threads per process.\n", size, num_threads);

        // Check the input file size
        if (stat(inFile, &inFileStat) == 0) {
            long inFileSize = inFileStat.st_size;
            printf("Reading from Input file: %s with size of %ld bytes.\n", inFile, inFileSize);
        } else {
            perror("Error getting input file size");
        }
    }

    // Read matrix and distribute among processes
    read_row_striped_matrix_halo(inFile, (void***)&matrix, MPI_DOUBLE, &rows, &cols, MPI_COMM_WORLD);
    exchange_row_striped_values((void***)&matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
    int local_rows = BLOCK_SIZE(rank, size, rows);

    // Cap threads
    int effective_threads = (local_rows < num_threads) ? local_rows : num_threads;
    omp_set_num_threads(effective_threads);

    if (size > rows) {
        fprintf(stderr, "Error: Number of processes (%d) cannot exceed the number of rows (%d).\n", size, rows);
        MPI_Finalize();
        return 1;
    }

    if (debug_level == 1) {
        printf("Process %d: Using %d threads for %d local rows (out of %d total rows).\n", rank, effective_threads, local_rows, rows);
    }

    // Work timing
    GET_TIME(work_start);


    read_row_striped_matrix_halo(inFile, (void***)&matrix1, MPI_DOUBLE, &rows, &cols, MPI_COMM_WORLD);

    if (rank == 0 && debug_level == 1) {
        printf("Starting stencil operation...\n");
    }

    // Hybrid MPI+OMP stencil computation
    #pragma omp parallel
    {
        for (int i = 0; i < iterations; i++) {
            // Apply stencil operation
            stencil2D_MPI_OMP(matrix, matrix1, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);

            // Ensure that only one thread handles file I/O
            #pragma omp single
            {
                if (allIterationsFile) {
                    GET_TIME(other_start);
                    {
                        if (i == 0) {
                            write_row_striped_matrix_halo(allIterationsFile, (void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
                        } else {
                            append_row_striped_matrix_halo(allIterationsFile, (void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
                        }
                    }
                    GET_TIME(other_end);

                    other_total += other_end - other_start;
                }

            }

            if (debug_level == 2) {
                #pragma omp single
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    print_row_striped_matrix_halo((void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
                }
            }

            // Swap pointers for the next iteration
            #pragma omp single
            {
                double **temp = matrix1;
                matrix1 = matrix;
                matrix = temp;
            }

            // Ensure all threads have updated pointers before communication
            #pragma omp barrier

            // Halo exchange for updated data
            #pragma omp single
            exchange_row_striped_values((void***)&matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
        }
    }

    if (allIterationsFile) {
        MPI_Barrier(MPI_COMM_WORLD);
        GET_TIME(other_start);
        append_row_striped_matrix_halo(allIterationsFile, (void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
        GET_TIME(other_end);

        other_total += other_end - other_start;
    }

    if (rank == 0 && debug_level == 1) {
        printf("Ending stencil operation...\n");
    }

    GET_TIME(work_end);

    // Write final result
    write_row_striped_matrix_halo(outFile, (void**)matrix, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);

    my_free((void **)matrix);
    my_free((void **)matrix1);

    // End overall timing
    GET_TIME(overall_end);

    // Timing calculations
    double overall_time = overall_end - overall_start;
    double work_time = work_end - work_start - other_total;
    double total_other_time = overall_time - work_time;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (debug_level >= 0) {
            printf("Time Overall: %.5f seconds\n", overall_time);
            printf("Time Computation: %.5f seconds\n", work_time);
            printf("Time Other: %.5f seconds\n", total_other_time);
        }
    }

    MPI_Finalize();
    return 0;
}
