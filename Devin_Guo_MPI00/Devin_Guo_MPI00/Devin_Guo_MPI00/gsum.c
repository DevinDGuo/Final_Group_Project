#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include "functions.h"

int main(int argc, char **argv) {
	int rank, size, value;
	double sum;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	value = rank;
	global_sum(&sum, rank, size, value);

	printf("FINAL IN MAIN: Process: %d has Sum = %f \n", rank, sum);

	MPI_Finalize();
	return 0;
}