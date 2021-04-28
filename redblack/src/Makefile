CC     =  gcc 
CLANG = clang
MCC    =  mcc
CFLAGS = -Wall -std=c99 -O0 -g -D_XOPEN_SOURCE=600
#CFLAGS = -Wall -std=c99 -O2 -ggdb3 -k -fno-inline
#SSGRIND_CFLAGS = -Wall -std=c99 -O0 -g
LIBS   = -lm
LIBSOMPSS =  


ALL	= heat gauss_omp 
all: $(ALL)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $< -o $@

heat: heat.c solver.c misc.o
	$(CC) $(CFLAGS) $+ $(LIBS) -o $@

gauss_omp: heat_omp.c gauss_omp.c misc.o
	$(CC) $(CFLAGS) -fopenmp $+ $(LIBS) -o $@


clean:
	rm -fr $(ALL) *.o *.xdot *.ppm *.out *.err *.prv *.pcf *.row TRACE.mpits set-0

