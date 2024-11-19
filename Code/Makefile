# Compiler and flags
CC = mpicc
CFLAGS = -g -Wall -Wstrict-prototypes -std=gnu99 -fopenmp
LFLAGS = -lm -lpthread

# Program names
PROGS = print-2d make-2d stencil-2d pth-stencil-2d omp-stencil-2d mpi-stencil-2d

# Object files
OBJS = utilities.o my_barrier.o

# All targets
all: $(PROGS)

# Link targets
print-2d: $(OBJS) print-2d.o
	$(CC) $(CFLAGS) -o print-2d $(OBJS) print-2d.o $(LFLAGS)

make-2d: $(OBJS) make-2d.o
	$(CC) $(CFLAGS) -o make-2d $(OBJS) make-2d.o $(LFLAGS)

stencil-2d: $(OBJS) stencil-2d.o
	$(CC) $(CFLAGS) -o stencil-2d $(OBJS) stencil-2d.o $(LFLAGS)

pth-stencil-2d: $(OBJS) pth-stencil-2d.o
	$(CC) $(CFLAGS) -o pth-stencil-2d $(OBJS) pth-stencil-2d.o $(LFLAGS)

omp-stencil-2d: $(OBJS) omp-stencil-2d.o
	$(CC) $(CFLAGS) -o omp-stencil-2d $(OBJS) omp-stencil-2d.o $(LFLAGS)

mpi-stencil-2d: $(OBJS) mpi-stencil-2d.o
	$(CC) $(CFLAGS) -o mpi-stencil-2d $(OBJS) mpi-stencil-2d.o MyMPI.c $(LFLAGS)

# Compile object files
utilities.o: utilities.c utilities.h my_barrier.h
	$(CC) $(CFLAGS) -c utilities.c -o utilities.o

my_barrier.o: my_barrier.c my_barrier.h
	$(CC) $(CFLAGS) -c my_barrier.c -o my_barrier.o

print-2d.o: print-2d.c utilities.h
	$(CC) $(CFLAGS) -c print-2d.c -o print-2d.o

make-2d.o: make-2d.c utilities.h
	$(CC) $(CFLAGS) -c make-2d.c -o make-2d.o

stencil-2d.o: stencil-2d.c utilities.h
	$(CC) $(CFLAGS) -c stencil-2d.c -o stencil-2d.o

pth-stencil-2d.o: pth-stencil-2d.c utilities.h
	$(CC) $(CFLAGS) -c pth-stencil-2d.c -o pth-stencil-2d.o

omp-stencil-2d.o: omp-stencil-2d.c utilities.h
	$(CC) $(CFLAGS) -c omp-stencil-2d.c -o omp-stencil-2d.o

mpi-stencil-2d.o: mpi-stencil-2d.c utilities.h
	$(CC) $(CFLAGS) -c mpi-stencil-2d.c -o mpi-stencil-2d.o
# Clean up
clean:
	rm -f *.o *.dat core* $(PROGS)
