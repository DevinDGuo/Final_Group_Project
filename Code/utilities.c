#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <mpi.h>
#include "timer.h"
#include "utilities.h"
#include "MyMPI.h"

// include "my_barrier.h"

void malloc2D(double ***x, int rows, int cols) {
    // First allocate a block of memory for the row pointers and the 2D array
    *x = (double **)malloc(rows * sizeof(double *) + rows * cols * sizeof(double));

    if (*x == NULL) {
        printf("Error: Memory allocation failed in malloc2D. \n");
        exit(1);
    }

    // Now assign the start of the block of memory for the 2D array after the row pointers
    (*x)[0] = (double *)(*x + rows);

    // Last, assign the memory location to point to for each row pointer
    for (int j = 1; j < rows; j++) {
        (*x)[j] = (*x)[j - 1] + cols;
    }
}

void fill2D(double** matrix, int rows, int cols) {
    if (matrix == NULL) {
        printf("Error: Invalid matrix pointer in fill2D.\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j == 0 || j == cols - 1) {
                matrix[i][j] = 1.00;  // Fill the first and last columns with 1.0
            } else {
                matrix[i][j] = 0.00;  // Fill other columns with 0.0
            }
        }
    }
}

// Function to write a 2D matrix to a file
int write2DMatrixToFile(double **matrix, int numRows, int numCols, const char *outputFile) {
    // Open the output file in binary mode
	FILE *outputFilePtr = fopen(outputFile, "wb");
	if (outputFilePtr == NULL) {
        printf("Error: Unable to open output file in write2DMatrixToFile.\n");
        exit(1);  
    }

    // Write the matrix dimensions to the output file
    fwrite(&numRows, sizeof(int), 1, outputFilePtr);
    fwrite(&numCols, sizeof(int), 1, outputFilePtr);

    // Write the matrix data to the output file row by row
    for (int i = 0; i < numRows; i++) {
    	fwrite(matrix[i], sizeof(double), numCols, outputFilePtr);
    }

    // Close the output file
    fclose(outputFilePtr);

    return 0;  // Return success code
}

int write2DMatrixToIterationFile(double **matrix, int numRows, int numCols, const char *outputFile) {
    // Open the output file in binary mode
	FILE *outputFilePtr = fopen(outputFile, "ab");
	if (outputFilePtr == NULL) {
        printf("Error: Unable to open output file in write2DMatrixToFile.\n");
        exit(1);  
    }

    // Write the full matrix to file instead of row by row
    fwrite(matrix[0], sizeof(double), numRows * numCols, outputFilePtr);

    // Close the output file
    fclose(outputFilePtr);

    return 0;  // Return success code
}

// Function to read a 2D matrix from a file
double **read2DMatrix(int *numRows, int *numCols, const char *inputFile) {
    // Open the input file
	FILE *inputFilePtr = fopen(inputFile, "rb");
	if (inputFilePtr == NULL) {
        printf("Error: Unable to open input file in read2DMatrix.\n");
        exit(1);
    }

    // Read matrix dimensions
	fread(numRows, sizeof(int), 1, inputFilePtr);
	fread(numCols, sizeof(int), 1, inputFilePtr);

    // Allocate memory for the matrix
    double **matrix;
	malloc2D(&matrix, *numRows, *numCols);
	if (matrix == NULL) {
		printf("Error: Memory allocation failed.\n");
		fclose(inputFilePtr);
		exit(1);
	}

    // Read the entire matrix data in one go
    fread(matrix[0], sizeof(double), (*numRows) * (*numCols), inputFilePtr);

    // Close the input file
	fclose(inputFilePtr);

	return matrix;
}

// Function to print a 2D matrix
void print2D(double **matrix, int numRows, int numCols) {
	if (matrix == NULL) {
        printf("Error: Invalid matrix pointer in print2D.\n");
        exit(1);  
    }

    for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			printf("%.2f ", matrix[i][j]);
		}
		printf("\n");
	}
}

void stencil2D(int numIterations, int rows, int cols, double **matrix, double **matrix1, const char *outputFile) {
    if (matrix == NULL || matrix1 == NULL) {
        printf("Error: Invalid matrix pointers in stencil2D.\n");
        exit(1);  
    }


    for (int iter = 0; iter < numIterations; iter++) {
        // Apply the stencil operation to the inner matrix cells only
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                // 9-point stencil calculation
                matrix1[i][j] = (matrix[i - 1][j - 1] + matrix[i - 1][j] + matrix[i - 1][j + 1] +
                                 matrix[i][j + 1] + matrix[i + 1][j + 1] + matrix[i + 1][j] +
                                 matrix[i + 1][j - 1] + matrix[i][j - 1] + matrix[i][j]) / 9.0;
            }
        }

        // Write the matrix to file at the appropriate iterations
        if (iter == 0) {
            write2DMatrixToFile(matrix, rows, cols, outputFile);  // Initial state
        } else {
            write2DMatrixToIterationFile(matrix, rows, cols, outputFile);  // Each subsequent iteration
        }

        // Swap matrices for the next iteration
        double **temp = matrix1;
        matrix1 = matrix;
        matrix = temp;
    }

    // Write the final matrix after all iterations
    write2DMatrixToIterationFile(matrix, rows, cols, outputFile);
}


// Thread function for performing stencil operations and optionally writing to file
void* stencil_thread_func(void* args) {
    ThreadArgs* threadArgs = (ThreadArgs*)args;
    int id = threadArgs->id;
    int rows = threadArgs->rows;
    int cols = threadArgs->cols;
    double **matrix = *(threadArgs->matrix);
    double **matrix1 = *(threadArgs->matrix1);
    int debugLevel = threadArgs->debugLevel;
    const char *outputFile = threadArgs->outputFile;
    double total_time = 0.0;

    // Calculate row range for each thread
    int startRow = BLOCK_LOW(id, threadArgs->numThreads, rows);
    int endRow = BLOCK_HIGH(id, threadArgs->numThreads, rows);

    // Adjust `startRow` and `endRow` to stay within bounds for stencil operations
    int actualStart = (startRow == 0) ? 1 : startRow;         // Avoid top row for stencil
    int actualEnd = (endRow == rows - 1) ? rows - 2 : endRow; // Avoid bottom row for stencil

    for (int iter = 0; iter < threadArgs->numIterations; iter++) {
        // Perform stencil operation within assigned rows
        for (int i = actualStart; i <= actualEnd; i++) {
            for (int j = 1; j < cols-1; j++) {
                    matrix1[i][j] = (matrix[i - 1][j - 1] + matrix[i - 1][j] + matrix[i - 1][j + 1] +
                                     matrix[i][j + 1] + matrix[i + 1][j + 1] + matrix[i + 1][j] +
                                     matrix[i + 1][j - 1] + matrix[i][j - 1] + matrix[i][j]) / 9.0;
            
            }
        }

        // Synchronize threads after processing each row block to ensure correct boundary values
        my_barrier_wait(threadArgs->barrier);

        // Only the main thread performs file writing, if applicable
        if (id == 0 && outputFile) {
            double other_start, other_end;
            GET_TIME(other_start);  // Start timing for file writing

            if (iter == 0) {
                write2DMatrixToFile(matrix, rows, cols, outputFile);
            } else {
                write2DMatrixToIterationFile(matrix, rows, cols, outputFile);
            }

            GET_TIME(other_end);  // End timing for file writing
            total_time += (other_end - other_start);  // Accumulate file writing time only
        }

        // Print matrix data for debugging if required
        if (id == 0 && debugLevel == 2) {
            printf("Iteration %d: \n", iter + 1);
            print2D(matrix, rows, cols);
        }

        // Swap matrix pointers for next iteration
        my_barrier_wait(threadArgs->barrier);  // Sync before swapping
        double **temp = matrix1;
        matrix1 = matrix;
        matrix = temp;
        my_barrier_wait(threadArgs->barrier);  // Sync after swapping
    }

    // Final file write after all iterations (main thread only)
    if (id == 0 && outputFile) {
        double other_start, other_end;
        GET_TIME(other_start);
        write2DMatrixToIterationFile(matrix, rows, cols, outputFile);
        GET_TIME(other_end);
        total_time += (other_end - other_start);
        *threadArgs->other_total = total_time;
    }

    // Ensure all threads complete before ending
    my_barrier_wait(threadArgs->barrier);

    return NULL;
}

void stencil2DPThread(int numIterations, int debugLevel, int rows, int cols, double **matrix, double **matrix1, 
                      const char *outputFile, double *other_total, int num_threads) {
    if (matrix == NULL || matrix1 == NULL) {
        if (debugLevel == 0) {
            printf("Error: Invalid matrix pointers in stencil2DPThread.\n");
        }
        exit(1); 
    }

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *threadArgs = malloc(num_threads * sizeof(ThreadArgs));
    my_barrier_t barrier;

    // Initialize the barrier for thread synchronization
    my_barrier_init(&barrier, 0, num_threads);

    // Initialize thread arguments and create threads
    for (int t = 0; t < num_threads; t++) {
        threadArgs[t].id = t;
        threadArgs[t].numThreads = num_threads;
        threadArgs[t].numIterations = numIterations;
        threadArgs[t].rows = rows;
        threadArgs[t].cols = cols;
        threadArgs[t].matrix = &matrix;   // Pass pointer to the matrix pointer
        threadArgs[t].matrix1 = &matrix1; // Pass pointer to the matrix1 pointer
        threadArgs[t].outputFile = outputFile;
        threadArgs[t].debugLevel = debugLevel;
        threadArgs[t].other_total = other_total;
        threadArgs[t].barrier = &barrier;
        pthread_create(&threads[t], NULL, stencil_thread_func, &threadArgs[t]);
    }

    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    // Clean up
    // pthread_barrier_destroy(&barrier);
    free(threads);
    free(threadArgs);
}

void stencil2DOMP(int numIterations, int debugLevel, int rows, int cols, double **matrix, double **matrix1, 
    const char *outputFile, double *other_total) {
    if (matrix == NULL || matrix1 == NULL) {
        if (debugLevel == 0) {
            printf("Error: Invalid matrix pointers in stencil2DPThread.\n");
        }
        exit(1); 
    }

    double other_start, other_end;

    for (int iter = 0; iter < numIterations; iter++) {
        #pragma omp parallel for
        for (int i = 1; i < rows-1; i++) {
            for (int j = 1; j < cols-1; j++) {
                    // Apply 9-point stencil operation
                    matrix1[i][j] = (matrix[i - 1][j - 1] + matrix[i - 1][j] + matrix[i - 1][j + 1] +
                                     matrix[i][j + 1] + matrix[i + 1][j + 1] + matrix[i + 1][j] +
                                     matrix[i + 1][j - 1] + matrix[i][j - 1] + matrix[i][j]) / 9.0;
    
            }
        }
        GET_TIME(other_start);
        if (outputFile) {
            if (iter == 0) {
                write2DMatrixToFile(matrix, rows, cols, outputFile);
            } else {
                write2DMatrixToIterationFile(matrix, rows, cols, outputFile);
            }
        }

        if (debugLevel == 2) {
            printf("Iteration %d: \n", iter+1);
            print2D(matrix, rows, cols);
        }
        GET_TIME(other_end);
        *other_total += (other_end - other_start);

        // Swap pointers directly
        double **temp = matrix1;
        matrix1 = matrix;
        matrix = temp;
    }
    GET_TIME(other_start);
    if (outputFile) {
        write2DMatrixToIterationFile(matrix, rows, cols, outputFile);
    }
    GET_TIME(other_end);
    *other_total += (other_end - other_start);
}

// Final Functions //

/*
 *   Function 'my_malloc' is called when a process wants
 *   to allocate some space from the heap. If the memory
 *   allocation fails, the process prints an error message
 *   and then aborts execution of the program.
 */

void my_free(void **matrix) {
    if (matrix != NULL) {
        free(matrix[0]); // Free the contiguous block of data
        free(matrix); // Free the array of pointers
    }
}

void *my_malloc (
   int id,     /* IN - Process rank */
   int bytes)  /* IN - Bytes to allocate */
{
   void *buffer;
   if ((buffer = malloc ((size_t) bytes)) == NULL) {
      printf ("Error: Malloc failed for process %d\n", id);
      fflush (stdout);
      MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
   }
   return buffer;
}

void my_allocate2d(int id, int local_rows, int n, int datum_size, void ***subs, void **storage) {
    // Allocate storage for the matrix
    *storage = (void *) my_malloc(id, local_rows * n * datum_size);
    *subs = (void **) my_malloc(id, local_rows * sizeof(void *)); 

    // Initialize the subs pointers
    void **lptr = (void **) *subs; 
    void *rptr = *storage;
    for (int i = 0; i < local_rows; i++) {
        lptr[i] = rptr; // Directly assign the pointer
        rptr += n * datum_size; // Move the pointer forward
    } 
}


void mpi_apply_stencil(double **matrix, double **matrix1, int rows, int cols) {
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            matrix1[i][j] = (matrix[i - 1][j - 1] + matrix[i - 1][j] + matrix[i - 1][j + 1] +
                             matrix[i][j - 1] + matrix[i][j + 1] +
                             matrix[i + 1][j - 1] + matrix[i + 1][j] + matrix[i + 1][j + 1] +
                             matrix[i][j]) / 9.0;
        }
    }
}

