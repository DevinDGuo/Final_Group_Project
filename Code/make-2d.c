#include <stdio.h>
#include <stdlib.h> 
#include "utilities.h" 
#include "timer.h" 

void printUsage(void) {
	printf("Usage: ./make-2d <rows> <cols> <output_file>\n");
}

int main(int argc, char* argv[]) {
	if (argc != 4) {
		printUsage();
		return 1;
	}

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);
	char *outFile = argv[3];
	double **matrix;

	malloc2D(&matrix, rows, cols);

	fill2D(matrix, rows, cols);

	write2DMatrixToFile(matrix, rows, cols, outFile);
	free(matrix);

	return 0;
}