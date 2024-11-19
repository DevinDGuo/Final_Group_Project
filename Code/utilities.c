#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "timer.h"
#include "utilities.h"
#include <mpi.h>

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

void my_free(void **matrix) {
    if (matrix != NULL) {
        free(matrix[0]); // Free the contiguous block of data
        free(matrix); // Free the array of pointers
    }
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

int get_size (MPI_Datatype t) {
   if (t == MPI_BYTE) return sizeof(char);
   if (t == MPI_DOUBLE) return sizeof(double);
   if (t == MPI_FLOAT) return sizeof(float);
   if (t == MPI_INT) return sizeof(int);
   printf ("Error: Unrecognized argument to 'get_size'\n");
   fflush (stdout);
   MPI_Abort (MPI_COMM_WORLD, TYPE_ERROR);

   return 0;
}

void print_submatrix (
   void       **a,       /* OUT - Doubly-subscripted array */
   MPI_Datatype dtype,   /* OUT - Type of array elements */
   int          rows,    /* OUT - Matrix rows */
   int          cols)    /* OUT - Matrix cols */
{
   int i, j;

   for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
         if (dtype == MPI_DOUBLE)
            printf ("%6.3f ", ((double **)a)[i][j]);
         else {
            if (dtype == MPI_FLOAT)
               printf ("%6.3f ", ((float **)a)[i][j]);
            else if (dtype == MPI_INT)
               printf ("%6d ", ((int **)a)[i][j]);
         }
      }
      putchar ('\n');
   }
}

void write_submatrix (
   const char *outputFileName,
   void       **a,       /* OUT - Doubly-subscripted array */
   MPI_Datatype dtype,   /* OUT - Type of array elements */
   int          rows,     /* OUT - Matrix rows */
   int          cols)     /* OUT - Matrix cols */
{

   FILE *outFile = fopen(outputFileName, "ab"); 
   if (outFile == NULL) {
      printf("Error: Unable to open output file for writing dimensions.\n");
      return;  
   }

   int i, j;
   for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
         if (dtype == MPI_DOUBLE) {
            fwrite(&((double **)a)[i][j], sizeof(double), 1, outFile);
         } else if (dtype == MPI_FLOAT) {
            fwrite(&((float **)a)[i][j], sizeof(float), 1, outFile);
         } else if (dtype == MPI_INT) {
            fwrite(&((int **)a)[i][j], sizeof(int), 1, outFile);
         }
      }
   }

   fclose(outFile); 
}

void read_row_striped_matrix_halo(
   char        *s,        /* IN - File name */
   void      ***subs,     /* OUT - 2D submatrix with halo rows */
   MPI_Datatype dtype,    /* IN - Matrix element type */
   int         *m,        /* OUT - Matrix rows */
   int         *n,        /* OUT - Matrix cols */
   MPI_Comm     comm)     /* IN - Communicator */
{
   int          datum_size;   /* Size of matrix element */
   int          i;
   int          id;           /* Process rank */
   FILE        *infileptr;    /* Input file pointer */
   int          local_rows;   /* Rows on this proc */
   int          p;            /* Number of processes */
   void        *storage;      /* Pointer for local storage, including halo */
   MPI_Status   status;       /* Result of receive */
   int          x;            /* Result of read */

   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &id);
   datum_size = get_size(dtype);

   if (id == (p-1)) {
      infileptr = fopen(s, "r");
      if (infileptr == NULL) *m = 0;
      else {
         fread(m, sizeof(int), 1, infileptr);
         fread(n, sizeof(int), 1, infileptr);
      }
   }

   MPI_Bcast(m, 1, MPI_INT, p-1, comm);
   if (!(*m)) MPI_Abort(MPI_COMM_WORLD, OPEN_FILE_ERROR);
   MPI_Bcast(n, 1, MPI_INT, p-1, comm);

   if (p > *m) {
      if (id == 0) {
         fprintf(stderr, "Error: Number of processes (%d) is greater than the number of rows (%d).\n", p, *m);
      }
      MPI_Finalize();
      exit(EXIT_FAILURE);
   }

   int halo_top, halo_bottom;

   if (id == 0) {
       // Process 0 has no top halo row
       halo_top = 0;
   } else {
       // Other processes have a top halo row
       halo_top = 1;
   }

   if (id == p - 1) {
       // The last process has no bottom halo row
       halo_bottom = 0;
   } else {
       // Other processes have a bottom halo row
       halo_bottom = 1;
   }

   local_rows = BLOCK_SIZE(id, p, *m) + halo_top + halo_bottom;

   // Dynamically allocate matrix with appropriate halo rows
   my_allocate2d(id, local_rows, *n, datum_size, subs, &storage);

   // Initialize halo rows with -1.000 if they exist
   if (halo_top) {
      for (i = 0; i < *n; i++) {
         ((double*)(*subs)[0])[i] = -1.0;  // Top halo row for non-zero processes
      }
   }
   if (halo_bottom) {
      for (i = 0; i < *n; i++) {
         ((double*)(*subs)[local_rows - 1])[i] = -1.0; // Bottom halo row for non-p-1 processes
      }
   }

   // Process p-1 reads blocks of rows from file and sends each block to the correct destination process.
   if (id == (p-1)) {
      for (i = 0; i < p - 1; i++) {
         x = fread(storage + datum_size * *n * halo_top, datum_size, BLOCK_SIZE(i, p, *m) * *n, infileptr);
         MPI_Send(storage + datum_size * *n * halo_top, BLOCK_SIZE(i, p, *m) * *n, dtype, i, DATA_MSG, comm);
      }
      x = fread(storage + datum_size * *n * halo_top, datum_size, (local_rows - halo_top - halo_bottom) * *n, infileptr);
      fclose(infileptr);
   } else {
      MPI_Recv(storage + datum_size * *n * halo_top, (local_rows - halo_top - halo_bottom) * *n, dtype, p-1, DATA_MSG, comm, &status);
   }
}


void print_row_striped_matrix_halo(
   void **a,            /* IN - 2D array with halos */
   MPI_Datatype dtype,  /* IN - Matrix element type */
   int m,               /* IN - Matrix rows */
   int n,               /* IN - Matrix cols */
   MPI_Comm comm)       /* IN - Communicator */
{
   MPI_Status  status;          /* Result of receive */
   void       *bstorage;        /* Elements received from another process */
   void      **b;               /* 2D array indexing into 'bstorage' */
   int         datum_size;      /* Bytes per element */
   int         i;
   int         id;              /* Process rank */
   int         local_rows;      /* This proc's rows */
   int         max_block_size;  /* Most matrix rows held by any process */
   int         prompt;          /* Dummy variable */
   int         p;               /* Number of processes */

   MPI_Comm_rank(comm, &id);
   MPI_Comm_size(comm, &p);

   int halo_top, halo_bottom;

   if (id == 0) {
       // Process 0 has no top halo row
       halo_top = 0;
   } else {
       // Other processes have a top halo row
       halo_top = 1;
   }

   if (id == p - 1) {
       // The last process has no bottom halo row
       halo_bottom = 0;
   } else {
       // Other processes have a bottom halo row
       halo_bottom = 1;
   }

   local_rows = BLOCK_SIZE(id, p, m) + halo_top + halo_bottom;

   if (!id) {
      printf("\n");
      // Print the submatrix with halos for process 0
      print_submatrix(a, dtype, local_rows, n);
      if (p > 1) {
         datum_size = get_size(dtype);
         max_block_size = BLOCK_SIZE(p - 1, p, m) + 2; 
         my_allocate2d(id, max_block_size, n, datum_size, &b, &bstorage);

         for (i = 1; i < p; i++) {
            MPI_Send(&prompt, 1, MPI_INT, i, PROMPT_MSG, MPI_COMM_WORLD);

            // Determine the number of halo rows for process i
            int halo_rows;
            if (i == p - 1) {
               halo_rows = 1;  // Last process only needs a top halo row
            } else {
               halo_rows = 2;  // Intermediate processes need both top and bottom halo rows
            }

            // Calculate the total number of rows (including halo rows) for process i
            int total_rows = BLOCK_SIZE(i, p, m) + halo_rows;

            // Receive data with the correct size including halo rows
            MPI_Recv(bstorage, total_rows * n, dtype, i, RESPONSE_MSG, MPI_COMM_WORLD, &status);

            // Print the submatrix including halo rows
            print_submatrix(b, dtype, total_rows, n);
         }


         free(b);
         free(bstorage);
      }
   } else {
      // Each process responds with its data, including halo rows if present
      MPI_Recv(&prompt, 1, MPI_INT, 0, PROMPT_MSG, MPI_COMM_WORLD, &status);
      MPI_Send(*a, local_rows * n, dtype, 0, RESPONSE_MSG, MPI_COMM_WORLD);
   }
}


void write_row_striped_matrix_halo(
   char *outputFile,
   void **a,
   MPI_Datatype dtype,
   int m,
   int n,
   MPI_Comm comm)
{
   int id, p, local_rows, datum_size, max_block_size;
   void *bstorage;
   void **b;
   MPI_Status status;
   int prompt;

   MPI_Comm_rank(comm, &id);
   MPI_Comm_size(comm, &p);

   local_rows = BLOCK_SIZE(id, p, m);

   if (id == 0) {
      FILE *outFile = fopen(outputFile, "wb");
      if (outFile == NULL) {
          printf("Error: Unable to open output file for writing dimensions.\n");
          return;  
      }
      fwrite(&m, sizeof(int), 1, outFile);
      fwrite(&n, sizeof(int), 1, outFile);
      fclose(outFile);
   }

   MPI_Barrier(comm);

   if (id == 0) {
      write_submatrix(outputFile, a, dtype, local_rows, n);

      if (p > 1) {
         datum_size = get_size(dtype);
         max_block_size = BLOCK_SIZE(p - 1, p, m);
         my_allocate2d(id, max_block_size, n, datum_size, &b, &bstorage);

         for (int i = 1; i < p; i++) {
            MPI_Send(&prompt, 1, MPI_INT, i, PROMPT_MSG, MPI_COMM_WORLD);
            MPI_Recv(bstorage, BLOCK_SIZE(i, p, m) * n, dtype, i, RESPONSE_MSG, MPI_COMM_WORLD, &status);
            write_submatrix(outputFile, b, dtype, BLOCK_SIZE(i, p, m), n);
         }
         free(b);
         free(bstorage);
      }
   } else {
      MPI_Recv(&prompt, 1, MPI_INT, 0, PROMPT_MSG, MPI_COMM_WORLD, &status);
      MPI_Send(a[1], local_rows * n, dtype, 0, RESPONSE_MSG, MPI_COMM_WORLD);
   }
}

void exchange_row_striped_values(void ***subs, MPI_Datatype dtype, int m, int n, MPI_Comm comm) {
   int id, p;

   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &id);

   MPI_Status status;

   int halo_top, halo_bottom;

   if (id == 0) {
      // Process 0 has no top halo row
      halo_top = 0;
   } else {
      // Other processes have a top halo row
      halo_top = 1;
   }

   if (id == p - 1) {
      // The last process has no bottom halo row
      halo_bottom = 0;
   } else {
      // Other processes have a bottom halo row
      halo_bottom = 1;
   }

   int local_rows = BLOCK_SIZE(id, p, m) + halo_top + halo_bottom;

   // This way, there wouldn't be a deadlock. Originally, I didn't have
   // this implemented this way, causing a daadlock. 

   // Even communicates with left, odd communicates with right
   if (id % 2 == 0 && (id - 1) >= 0) {
      MPI_Sendrecv(
         (*subs)[1],              // Send second row
         n, dtype,                // Dimensions and type
         id - 1, 1,               // Destination: left neighbor
         (*subs)[0],              // Receive top halo row
         n, dtype,                // Dimensions and type
         id - 1, 1,               // Source: left neighbor
         comm, &status            // Communicator and status
         );
   } else if (id % 2 == 1 && (id + 1) < p) {
      MPI_Sendrecv(
         (*subs)[local_rows - 2], // Send second-to-last row
         n, dtype,                // Dimensions and type
         id + 1, 1,               // Destination: right neighbor
         (*subs)[local_rows - 1], // Receive bottom halo row
         n, dtype,                // Dimensions and type
         id + 1, 1,               // Source: right neighbor
         comm, &status            // Communicator and status
         );
   }

   // Even communicates with right, odd communicates with 
   if (id % 2 == 0 && (id + 1) < p) {
      MPI_Sendrecv(
         (*subs)[local_rows - 2], // Send second-to-last row
         n, dtype,                // Dimensions and type
         id + 1, 0,               // Destination: right neighbor
         (*subs)[local_rows - 1], // Receive bottom halo row
         n, dtype,                // Dimensions and type
         id + 1, 0,               // Source: right neighbor
         comm, &status            // Communicator and status
         );
   } else if (id % 2 == 1 && (id - 1) >= 0) {
      MPI_Sendrecv(
         (*subs)[1],              // Send second row
         n, dtype,                // Dimensions and type
         id - 1, 0,               // Destination: left neighbor
         (*subs)[0],              // Receive top halo row
         n, dtype,                // Dimensions and type
         id - 1, 0,               // Source: left neighbor
         comm, &status            // Communicator and status
         );
   }
}

