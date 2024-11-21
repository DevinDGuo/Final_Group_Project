#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <mpi.h>

char* toBinary(int n, int width) {
    char* binaryStr = (char*)malloc((width + 1) * sizeof(char));

    for (int i = width - 1; i >= 0; i--) {
        binaryStr[i] = (n % 2) + '0';
        n = n / 2;
    }
    binaryStr[width] = '\0';

    return binaryStr;
}

void global_sum(double* result, int rank, int size, double my_value) {
	if ((size & (size-1)) != 0) {
		perror("Size is not a power of two.");
        exit(1);
	};
	for (int i=0; i<log2(size); i++) {
		int mask = 1<<i;
		int xrank = rank^mask;
		double my_other_value;

		if ((rank & mask) == 0) {
			MPI_Ssend(&my_value, 1, MPI_DOUBLE, xrank, 2, MPI_COMM_WORLD);
			MPI_Recv(&my_other_value, 1, MPI_DOUBLE, xrank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&my_other_value, 1, MPI_DOUBLE, xrank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Ssend(&my_value, 1, MPI_DOUBLE, xrank, 2, MPI_COMM_WORLD);
		}

		char* binary1 = toBinary(rank, log2(size));
		char* binary2 = toBinary(xrank, log2(size));
		printf(" Phase %d - P %d (%s) receiving from P %d (%s), val %.1f\n", i, rank, binary1, xrank, binary2, my_other_value);
		printf(" Phase %d - P %d (%s) sending to     P %d (%s), val %.1f\n", i, rank, binary1, xrank, binary2,my_value);

		my_value += my_other_value;
	}
	*result = my_value;
};



