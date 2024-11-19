#ifndef UTILITIES_H
#define UTILITIES_H

#include "my_barrier.h"

// Quinn Macros
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)

typedef struct {
    int id;
    int numThreads;
    int numIterations;
    int rows;
    int cols;
    double ***matrix;
    double ***matrix1;
    const char *outputFile;
    int debugLevel;
    double *other_total;
    my_barrier_t *barrier;
} ThreadArgs;

void fill2D(double **matrix, int rows, int cols);

void printUsage(void);
// Function to dynamically allocate memory for a 2D array
void malloc2D(double ***x, int rows, int cols);

// Function to write a 2D matrix to a file
int write2DMatrixToFile(double **matrix, int numRows, int numCols, const char *outputFile);

int write2DMatrixToIterationFile(double **matrix, int numRows, int numCols, const char *outputFile);

// Function to read a 2D matrix from a file
double **read2DMatrix(int *numRows, int *numCols, const char *inputFile);

// Function to print a 2D matrix
void print2D(double **matrix, int numRows, int numCols);

void stencil2D(int numIterations, int rows, int cols, double **matrix, double **matrix1, const char *outputFile);

void* stencil_thread_func(void* args);

void stencil2DPThread(int numIterations, int debugLevel, int rows, int cols, double **matrix, double **matrix1, 
	const char *outputFile, double *other_total, int num_threads);

void stencil2DOMP(int numIterations, int debugLevel, int rows, int cols, double **matrix, double **matrix1, const char *outputFile, double *other_total);

#endif /* UTILITIES_H */