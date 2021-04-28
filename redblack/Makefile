CC     =  gcc 
CLANG = clang
MCC    =  mcc
CFLAGS = -Wall -std=c99 -O0 -g -D_XOPEN_SOURCE=600
#CFLAGS = -Wall -std=c99 -O2 -ggdb3 -k -fno-inline
#SSGRIND_CFLAGS = -Wall -std=c99 -O0 -g
LIBS   = -lm
LIBSOMPSS =  


ALL	= redblack_serial redblack_omp
all: $(ALL)

misc.o: ./src/misc.c
	$(CC) -c $(CFLAGS) $< -o $@

redblack_serial: ./src/heat.c ./src/solver.c misc.o
	$(CC) $(CFLAGS) $+ $(LIBS) -o $@

redblack_omp: ./src/heat_omp.c ./src/redblack_omp.c misc.o
	$(CC) $(CFLAGS) -fopenmp $+ $(LIBS) -o $@


clean:
	rm -fr $(ALL) *.o *.xdot *.out *.err *.prv *.pcf *.row TRACE.mpits set-0

