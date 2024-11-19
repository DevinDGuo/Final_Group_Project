#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include "MyMPI.h"
#include "utilities.h"
#define dtype double
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if(argc!=4) {
        printf("usage: mpirun -np <p> %s <in data file> <out data file> <out data file from halo array> \n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    
    //void read_row_striped_matrix (
    //   char        *s,        /* IN - File name */
    //   void      ***subs,     /* OUT - 2D submatrix indices */
    //   MPI_Datatype dtype,    /* IN - Matrix element type */
    //   int         *m,        /* OUT - Matrix rows */
    //   int         *n,        /* OUT - Matrix cols */
    //   MPI_Comm     comm)     /* IN - Communicator */
    
    dtype** A; int rows, cols;
    

    read_row_striped_matrix(argv[1], (void***)&A, MPI_DOUBLE, &rows, &cols, MPI_COMM_WORLD);
    
    //void print_row_striped_matrix (
    //   void **a,            /* IN - 2D array */
    //   MPI_Datatype dtype,  /* IN - Matrix element type */
    //   int m,               /* IN - Matrix rows */
    //   int n,               /* IN - Matrix cols */
    //   MPI_Comm comm)       /* IN - Communicator */
    
    // just print out the matrix, showing the Quinn's version will work
    print_row_striped_matrix((void**)A, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
    
    //void write_row_striped_matrix (
    //   char* outFile,       /* IN - output file name */
    //   void **a,            /* IN - 2D array */
    //   MPI_Datatype dtype,  /* IN - Matrix element type */
    //   int m,               /* IN - Matrix rows */
    //   int n,               /* IN - Matrix cols */
    //   MPI_Comm comm)       /* IN - Communicator */
    
    // now, let's try to write out the matrix, using our new function, that is based on Quinn's print version.
    write_row_striped_matrix(argv[2], (void**)A, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);
    
    // now, let's make a new matrix in memory that is the 'halo' version of the matrix
    dtype** A2; int rows2, cols2;
    
    // read in the matrix.  the matrix we're reading is is the same format, but then it allocates the matrix,
    // the matrix will have the extra rows (depending on which process it is), and it will read and place the matrix
    // where it needs to be.  so the rows and cols are the same as before, but internally, it will need extra rows
    // make this function using his read() as a starting place
    read_row_striped_matrix_halo(argv[1], (void***)&A2, MPI_DOUBLE, &rows2, &cols2, MPI_COMM_WORLD);
    
    // print out the actual matrix in memory, which includes the halos, so that we can see that it is there.
    print_row_striped_matrix_halo((void**)A2, MPI_DOUBLE, rows2, cols2, MPI_COMM_WORLD);

    // Perform the exchange
    exchange_row_striped_values((void***)&A2, MPI_DOUBLE, rows2, cols2, MPI_COMM_WORLD);

    print_row_striped_matrix_halo((void**)A2, MPI_DOUBLE, rows2, cols2, MPI_COMM_WORLD);

    // then, given that A2 is a halo version of the matrix, write the data to the file
    // but the data in the file, will not have the halo information, so that it will match the
    // format that we need for the final state of the 2d heat plate.
    write_row_striped_matrix_halo(argv[3], (void**)A2, MPI_DOUBLE, rows, cols, MPI_COMM_WORLD);

    my_free((void**)A);
    my_free((void**)A2);

    
    MPI_Finalize();
    return 0;
}
