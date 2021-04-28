#include "heat-ompss.h"
#include <stdlib.h>
#include <omp.h>

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

extern int Task_Cntr_Out;
/*
 * Blocked Jacobi solver: one iteration step
 */
void relax_jacobi (unsigned padding, unsigned sizey, double (*u)[sizey+padding*2], double (*utmp)[sizey+padding*2], unsigned sizex, int check, double *residual )
{
    int nbx, bx, nby, by;
    int inf_i, sup_i, inf_j, sup_j;
  
    nbx = 2;
    bx = sizex/nbx;
    nby = 2;
    by = sizey/nby;
    
    double *sum;
    double (*local_sum)[nby];
    posix_memalign( (void**)&local_sum, sizeof( double) * nbx *nby, sizeof( double) * nbx *nby );
    
    if ( check ) {
        posix_memalign( (void**)&sum, sizeof( double ), sizeof( double ) );
        *sum=0.0;
    }

    
    #pragma omp parallel num_threads(4) 
    {
    #pragma omp single  nowait //
    {

    for (int ii=padding+1; ii<sizex-1+padding; ii+=bx) 
    {
        for (int jj=padding+1; jj<sizey-1+padding; jj+=by) 
        {
            inf_i = ii; inf_j=jj; 
            sup_i = min( ii+bx, sizex-1 + padding );
            sup_j = min( jj+by, sizey-1 + padding );
                 

            #pragma omp task depend( in:\
                    u[inf_i-1][inf_j:sup_j-1],\
                    u[sup_i][inf_j:sup_j-1],\
                    u[inf_i:sup_i-1][inf_j-1],\
                    u[inf_i:sup_i-1][sup_j],\
                    u[inf_i:sup_i-1][inf_j:sup_j-1] )\
                depend( out: utmp[inf_i:sup_i-1][inf_j:sup_j-1],\
                    local_sum[(ii-padding)/bx][(jj-padding)/by] ) 
            {
                if ( check ) local_sum[(ii-padding)/bx][(jj-padding)/by]=0.0f;
            
                for (int i=inf_i; i<sup_i; i++) {
                    for (int j=inf_j; j<sup_j; j++) 
                    {
                        double diff;
                        utmp[i][j]= 0.25 * (u[ i][j-1 ]+  // left
                                                 u[ i][(j+1) ]+  // right
                                                 u[(i-1)][ j]+  // top
                                                 u[ (i+1)][ j ]); // bottom
                        if ( check ) 
                        {
                            diff = utmp[i][j] - u[i][j];
                            local_sum [(ii-padding)/bx][(jj-padding)/by] += diff * diff;
                        }
                    }
                }
                
            }

            if ( check ) 
            {
                #pragma omp task depend( in :local_sum [(ii-padding)/bx][(jj-padding)/by]) depend( inout: sum ) //label(sum_acc)
                {
                    *sum += local_sum [(ii-padding)/bx][(jj-padding)/by];
                }
            }
        }
    }


    
    if ( check ) 
    {
            #pragma omp task depend( in : sum ) depend( out : residual ) //label(residual_set)
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
}  // end omp single
} // end omp parallel

}

