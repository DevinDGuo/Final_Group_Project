#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"
#include "timer.h"

void printUsage() {
    printf("Usage: ./stencil-2d <num iterations> <input file> <output file> <all-iterations>\n");
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printUsage();
        return 1;
    }

    double overall_start, overall_end, work_start, work_end;

    // Start overall timing
    GET_TIME(overall_start);

    int iterations = atoi(argv[1]);
    char *inFile = argv[2];
    char *outFile = argv[3];
    double **matrix1;
    char *allIterationsFile = argv[4];

    int rows, cols;

    double **matrix = read2DMatrix(&rows, &cols, inFile);
    if (matrix == NULL) {
        printf("Error: Failed to read matrix from file.\n");
        return 1;
    }

    malloc2D(&matrix1, rows, cols);

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

    stencil2D(iterations, rows, cols, matrix, matrix1, allIterationsFile);

    GET_TIME(work_end);

    write2DMatrixToFile(matrix, rows, cols, outFile);

    free(matrix);
    free(matrix1);

    // End overall timing
    GET_TIME(overall_end);

    // Calculate times
    double overall_time = overall_end - overall_start;
    double work_time = work_end - work_start;
    double io_time = overall_time - work_time;

    printf("Overall Time: %.5f seconds\n", overall_time);
    printf("Work Time: %.5f seconds\n", work_time);
    printf("Other Time: %.5f seconds\n", io_time);

    return 0;
}