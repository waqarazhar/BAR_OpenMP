#include "heat-ompss.h"
#include <stdlib.h>

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
void relax_gauss (unsigned padding, unsigned sizey, double (*u)[sizey+padding*2], unsigned sizex, int check, double *residual )
{
    // gmiranda: Do not use variables in the stack for tasks since they might be destroyed
    double *sum;

    int nbx, bx, nby, by;
    int inf_i, sup_i, inf_j, sup_j;

    nbx = 2;
    bx = sizex/nbx;
    nby = 2;
    by = sizey/nby;
    
    //extern int Task_Cntr_Out;

    // gmiranda: same, stack vars are not allowed!
    double (*local_sum)[nby];
    posix_memalign( (void**)&local_sum, sizeof( double) * nbx *nby, sizeof( double) * nbx *nby );

    if ( check ) {
        posix_memalign( (void**)&sum, sizeof( double ), sizeof( double ) );
        //#pragma omp task out (*sum) label(init_sum)
        *sum=0.0;
    }
   	
    
    static int Task_Cntr_Out;
   	#pragma omp parallel num_threads(4) 
  	{
  	#pragma omp single  nowait //
  	{	
    	for (int ii=padding+1; ii<sizex-1+padding; ii+=bx)
    	{
        	for (int jj=padding+1; jj<sizey-1+padding; jj+=by) 
        	{
				inf_i = ii; inf_j=jj; 
	 			sup_i = (ii+bx)<sizex-1 + padding ? ii+bx : sizex-1 + padding; 	
	 			sup_j = (jj+bx)<sizey-1 + padding ? jj+by : sizey-1 + padding;
				// north, south, west, east		

				#pragma omp task depend(\
					in: u[inf_i-1][inf_j:sup_j-1],\
   					u[sup_i][inf_j:sup_j-1],\
    				u[inf_i:sup_i-1][inf_j-1],\
    				u[inf_i:sup_i-1][sup_j] )\
    				depend( inout: u[inf_i:sup_i-1][inf_j:sup_j-1] )\
    				depend( out: local_sum[(ii-padding)/bx][(jj-padding)/by] ) // label(gauss_iteration)
				{
            	if ( check )
                local_sum[(ii-padding)/bx][(jj-padding)/by]=0.0f;
	
            	for (int i=inf_i; i<sup_i; i++)
                	for (int j=inf_j; j<sup_j; j++) 
                	{
                    	double unew, diff;
                    	unew= 0.25 * (    u[i][(j-1)]+  // left
                          u[i][(j+1)]+  // right
                          u[(i-1)][j]+  // top
                          u[(i+1)][j]); // bottom
                    	if ( check ) 
                    	{
                        	diff = unew - u[i][j];
                        	local_sum [(ii-padding)/bx][(jj-padding)/by] += diff * diff;
                    	}
                    	u[i][j]=unew;
                	}
            	}
            
        		if ( check ) 
        		{
            		#pragma omp task depend( in : local_sum[(ii-padding)/bx][(jj-padding)/by] ) depend( inout: sum ) //shared(sum) //label(sum_acc)
            		{	
                		*sum += local_sum [ (ii-padding)/bx ][ (jj-padding)/by ];
            		}
        		}

        	}
    	}

    
    if ( check ) 
    {
        #pragma omp task depend( in:sum) depend( out: residual) // label(residual_set)
        {
            *residual = *sum;
            // Release local variables
            free( local_sum );
            free( sum );
        }
    }
    else 
    {
        // Free local_sum as it will not be released if check is false
        free( local_sum );
    }

	} // end Omp single
	} // end omp parallel
}
