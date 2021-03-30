/*
 * Iterative solver for heat distribution
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "heat-ompss.h"
#include <omp.h>

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;

    // algorithmic parameters
    algoparam_t param;
    int np;

    double runtime, flop;
    double residual=0.0;
    int checkresid;

    // check arguments
    if( argc < 2 )
    {
	usage( argv[0] );
	return 1;
    }

    // check input file
    if( !(infile=fopen(argv[1], "r"))  ) 
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
      
	usage(argv[0]);
	return 1;
    }

    // check result file
    resfilename= (argc>=3) ? argv[2]:"heat.ppm";

    if( !(resfile=fopen(resfilename, "w")) )
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for writing.\n\n", 
		resfilename);
	usage(argv[0]);
	return 1;
    }

    // check input
    if( !read_input(infile, &param) )
    {
	fprintf(stderr, "\nError: Error parsing input file.\n\n");
	usage(argv[0]);
	return 1;
    }
    print_params(&param);

    if( !initialize(&param, 0) )
    {
	fprintf(stderr, "Error in Solver initialization.\n\n");
	usage(argv[0]);
	return 1;
    }

    // full size (param.resolution are only the inner points)
    np = param.resolution + 2;
    
    // starting time
    runtime = wtime();
    iter = 0;
    while(1) 
    {
		if ( ((iter%1000)==0) || (iter==param.maxiter-1) ) checkresid=1;
		else checkresid=0;
		
		
				relax_gauss(param.padding, np, (double (*)[np+2*param.padding])param.u, np, checkresid, &residual);	
		//	}
		//}		
        iter++;

        // solution good enough ?
		if (checkresid)
		{
	    	#pragma omp taskwait 
	    	if (residual < 0.00005) break;
		}

        // max. iteration reached ? (no limit with maxiter=0)
        if (param.maxiter>0 && iter>=param.maxiter) break;
    }

    #pragma omp taskwait
    // Flop count after iter iterations
    flop = iter * 11.0 * param.resolution * param.resolution;
    // stopping time
    runtime = wtime() - runtime;

    fprintf(stdout, "Time: %04.3f ", runtime);
    fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
	flop/1000000000.0,
	flop/runtime/1000000);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

    // for plot...
    coarsen( param.u, np, np, param.padding,param.uvis, param.visres+2, param.visres+2 );
  
    write_image( resfile, param.uvis, param.padding,param.visres+2,param.visres+2);

    finalize( &param );

    return 0;
}
