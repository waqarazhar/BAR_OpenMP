#all: sparselu
# this is a modified file for gcc or clang

CC = gcc

Exe_name=sparselu20
sparselu:  ./src/sparselu.c
	$(CC) -g -fopenmp ./src/sparselu.c -o $(Exe_name)

clean:  
	rm -Rf $(Exe_name) 
