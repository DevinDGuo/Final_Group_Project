#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

void printUsage() {
	printf("Usage: ./print-2d <input data file>\n");
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		printUsage();
		return 1;
	}

	char *inFile = argv[1];

	int rows; 
	int cols;

    double **matrix = read2DMatrix(&rows, &cols, inFile); 
    if (matrix == NULL) {
        printf("Error: Failed to read matrix from file.\n");
        return 1; 
    }

    printf("Matrix read from file:\n");
    print2D(matrix, rows, cols); 

    free(matrix);

    return 0;
}


