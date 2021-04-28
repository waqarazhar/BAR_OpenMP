/*
* Copyright (c) 2013, BSC (Barcelona Supercomputing Center)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the <organization> nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY BSC ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// modified version for openMP 4.0


#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <omp.h>

#include <sys/time.h>
#include <time.h>


#define NB 20
#define BSIZE 100  //100
#define FALSE (0)
#define TRUE (1)

typedef double ELEM;

ELEM *A[NB][NB];


void genmat ()
{
   int init_val, i, j, ii, jj;
   ELEM *p;

   init_val = 1325;

   for (ii = 0; ii < NB; ii++) 
     for (jj = 0; jj < NB; jj++)
     {
        p = A[ii][jj];
        if (p!=NULL)
           for (i = 0; i < BSIZE; i++) 
              for (j = 0; j < BSIZE; j++) {
	           init_val = (3125 * init_val) % 65536;
      	           (*p) = (ELEM)((init_val - 32768.0) / 16384.0);
                   p++;
              }
     }
}

void  print_structure()
{
   ELEM *p;
   int sum = 0; 
   int ii, jj, i, j;

   printf ("Structure for matrix A\n");

   for (ii = 0; ii < NB; ii++) {
     for (jj = 0; jj < NB; jj++) {
        p = A[ii][jj];
        if (p!=NULL)
        {
           {
              for(i =0; i < BSIZE; i++)
              {
                 for(j =0; j < BSIZE; j++)
                 {
                    printf("%+lg ", p[i * BSIZE + j]);
                 }
                 printf("\n");
              }
           }
        }
     }
   }
}

ELEM *allocate_clean_block()
{
  int i,j;
  ELEM *p, *q;

  p=(ELEM*)malloc(BSIZE*BSIZE*sizeof(ELEM));
  q=p;
  if (p!=NULL){
     for (i = 0; i < BSIZE; i++) 
        for (j = 0; j < BSIZE; j++){(*p)=(ELEM)0.0; p++;}
	
  }
  else printf ("OUT OF MEMORY!!!!!!!!!!!!!!!\n");
  return (q);
}

/* ************************************************************ */
/* Utility routine to measure time                              */
/* ************************************************************ */

double myusecond()
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return ((double) tv.tv_sec *4000000) + tv.tv_usec;
}

double mysecond()
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return ((double) tv.tv_sec) + ((double)tv.tv_usec*1e-6);
}

double gtod_ref_time_sec = 0.0;

float get_time()
{
    double t, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    // If this is the first invocation of through dclock(), then initialize the
    // "reference time" global variable to the seconds field of the tv struct.
    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double) tv.tv_sec;

    // Normalize the seconds field of the tv struct so that it is relative to the
    // "reference time" that was recorded during the first invocation of dclock().
    norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

    // Compute the number of seconds since the reference time.
    t = norm_sec + tv.tv_usec * 1.0e-6;

    return (float) t;
}

/* ************************************************************ */
/* main 							*/
/* ************************************************************ */

int main(int argc, char* argv[])
{
   float t_start,t_end;
   float time;
   int ii, jj, kk;
   int null_entry;
   int bcount = 0;

   //#pragma css start
   for (ii=0; ii<NB; ii++)
      for (jj=0; jj<NB; jj++)
      {
         null_entry=FALSE;
         if ((ii<jj) && (ii%3 !=0)) null_entry =TRUE;
         if ((ii>jj) && (jj%3 !=0)) null_entry =TRUE;
         if (ii%2==1) null_entry=TRUE;
         if (jj%2==1) null_entry=TRUE;
         if (ii==jj) null_entry=FALSE;
         if (null_entry==FALSE)
         {
            //printf("ii:%d jj:%d \n", ii,jj);
            bcount++;
            A[ii][jj] = (ELEM *)malloc(BSIZE*BSIZE*sizeof(ELEM));
            if (A[ii][jj]==NULL) 
            {
               printf("Out of memory\n");
               exit(1);
            }
         }
         else A[ii][jj] = NULL;
      }

   genmat();
   //print_structure();
   printf("Init OK Matrix is: %d (%d %d) # of blocks: %d memory is %ld MB\n", (NB*BSIZE), NB, BSIZE, bcount, bcount*sizeof(ELEM)/1024/1024);

   t_start=get_time();
 

for (kk=0; kk<NB; kk++) 
{
      ELEM *diag = A[kk][kk];
  
  #pragma omp parallel num_threads(1) 
  {
  #pragma omp single  nowait //
  {
	  #pragma omp task  depend(inout: diag[:100*100] ) 
      {
         int i, j, k;

         for (k=0; k<BSIZE; k++)
            for (i=k+1; i<BSIZE; i++) 
            {
               diag[i*BSIZE+k] = diag[i*BSIZE+k] / diag[k*BSIZE+k];
               for (j=k+1; j<BSIZE; j++)
                  diag[i*BSIZE+j] = diag[i*BSIZE+j] - diag[i*BSIZE+k] * diag[k*BSIZE+j];
            }
      }

      #pragma omp taskwait

      for (jj=kk+1; jj<NB; jj++)
      {
         if (A[kk][jj] != NULL)
         {
          
            ELEM *diag = A[kk][kk];
            ELEM *col = A[kk][jj];
			       #pragma omp task  depend( in:diag[:100*100] ) depend(inout: col[:100*100] ) 
            {
               int i, j, k;
               for (k=0; k<BSIZE; k++) 
                  for (i=k+1; i<BSIZE; i++)
                     for (j=0; j<BSIZE; j++)
                        col[i*BSIZE+j] = col[i*BSIZE+j] - diag[i*BSIZE+k]*col[k*BSIZE+j];
            }
         }
       }

      	for (ii=kk+1; ii<NB; ii++) 
      	{
         	if (A[ii][kk] != NULL)
         	{
            	ELEM *row = A[kk][kk];
            	ELEM *diag = A[ii][kk];
				      #pragma omp task  depend(in: diag[:100*100] ) depend( inout: row[:100*100] )
            	{
               		int i, j, k;

               		for (i=0; i<BSIZE; i++)
                  		for (k=0; k<BSIZE; k++) 
                  		{
                     		row[i*BSIZE+k] = row[i*BSIZE+k] / diag[k*BSIZE+k];
                     		for (j=k+1; j<BSIZE; j++)
                        		row[i*BSIZE+j] = row[i*BSIZE+j] - row[i*BSIZE+k]*diag[k*BSIZE+j];
                  		}
            	}
            
            for (jj=kk+1; jj<NB; jj++)
            {
               if (A[kk][jj] != NULL)
               {
                  if (A[ii][jj]==NULL)
                  {
                     A[ii][jj]=allocate_clean_block();
                  }
                  {

                     ELEM *row = A[ii][kk];
                     ELEM *col = A[kk][jj];
                     ELEM *inner = A[ii][jj];
                      #pragma omp task depend(in: row[:100*100], col[:100*100]) depend( inout: inner[:100*100] )          
                     {
                       
                        int i, j, k;

                        for (i=0; i<BSIZE; i++){
                              for (k=0; k<BSIZE; k++) {
                           		for (j=0; j<BSIZE; j++){
                                 inner[i*BSIZE+j] = inner[i*BSIZE+j] - row[i*BSIZE+k]*col[k*BSIZE+j];
                              }
                           }
                        }
                     }


                  }
               }
            }
         }
      }
   } }  // end of omp parallel

 //#pragma omp taskwait // added be waqar
   
}

#pragma omp taskwait
   t_end=get_time();

   time = t_end-t_start;
   printf("Matrix is: %d (%d %d) memory is %ld MB time to compute = %11.4f sec\n", 
         (NB*BSIZE), NB, BSIZE, (NB*BSIZE)*(NB*BSIZE)*sizeof(ELEM)/1024/1024, time);
   //print_structure();
}

