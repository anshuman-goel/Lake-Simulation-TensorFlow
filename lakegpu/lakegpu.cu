/******************************************************************************
* FILE: lakegpu.cu
*
* Group Info:
* agoel5 Anshuman Goel
* kgondha Kaustubh Gondhalekar
* ndas Neha Das
*
* LAST REVISED: 9/19/2017
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define VSQR 0.1
#define TSCALE 1.0

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/

extern int tpdt(double *t, double dt, double end_time);

__device__ double fn(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

//updates the grid state from time t to time t+dt
__global__ static void evolve(double *un, double *uc, double *uo, double *pebbles, int *n, double *h, double *dt, double *t, int *n_blocks, int *n_threads)
{
  int i, j;
  const unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int gid = idx;

  //assuming threads*blocks is not a power of 2, 
  //the leftover variable stores the excess number of grid points we need to compute
  int leftover = (*n * *n) - (*n_threads * *n_blocks);

  i = idx / *n;
  j = idx % *n;

  //values at lake edge points are set to zero
  if( i == 0 || i == *n - 1 || j == 0 || j == *n - 1 ||
      i == *n - 2 || i == 1 || j == *n - 2 || j == 1)
  {
    un[idx] = 0.;
  }
  else
  {
    //compute the 13-point stencil function for every grid point
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(*dt * *dt) *
    ((  uc[idx-1] // WEST
        + uc[idx+1] // EAST
        + uc[idx + *n] // SOUTH
        + uc[idx - *n] // NORTH
       + 0.25*( uc[idx - *n - 1 ] // NORTHWEST
              + uc[idx - *n + 1 ] // NORTHEAST
              + uc[idx + *n - 1 ] // SOUTHWEST
              + uc[idx + *n + 1 ] // SOUTHEAST
              )
      + 0.125*( uc[idx - 2 ]  // WESTWEST
              + uc[idx + 2 ] // EASTEAST
              + uc[idx - 2 * *n ] // NORTHNORTH
              + uc[idx + 2 * *n ] // SOUTHSOUTH
              )
      - 6 * uc[idx])/(*h * *h) + fn(pebbles[idx],*t));
    }

    //thread 0 in the last block handles computation for the leftover grid points
    if (blockIdx.x == *n_blocks - 1 && tid == 0 && leftover > 0)
    {
      for( idx = *n * *n - 1; idx>= *n * *n - leftover; idx--)
      {
        i = idx / *n;
        j = idx % *n;
        if( i == 0 || i == *n - 1 || j == 0 || j == *n - 1 ||
            i == *n - 2 || i == 1 || j == *n - 2 || j == 1)
        {
          un[idx] = 0.;
        }
        else
        {

          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(*dt * *dt) *
          ((  uc[idx-1] // WEST
              + uc[idx+1] // EAST
              + uc[idx + *n] // SOUTH
              + uc[idx - *n] // NORTH
             + 0.25*( uc[idx - *n - 1 ] // NORTHWEST
                    + uc[idx - *n + 1 ] // NORTHEAST
                    + uc[idx + *n - 1 ] // SOUTHWEST
                    + uc[idx + *n + 1 ] // SOUTHEAST
                    )
            + 0.125*( uc[idx - 2 ]  // WESTWEST
                    + uc[idx + 2 ] // EASTEAST
                    + uc[idx - 2 * *n ] // NORTHNORTH
                    + uc[idx + 2 * *n ] // SOUTHSOUTH
                    )
            - 6 * uc[idx])/(*h * *h) + fn(pebbles[idx],*t));
          }

      }
    }
    __syncthreads();

    //save most recent two time-stamps into uo and uc
    uo[gid] = uc[gid];
    uc[gid] = un[gid];
    //update leftover grid point's timestamps
    if (blockIdx.x == *n_blocks - 1 && tid == 0 && leftover > 0)
    {
      for( idx = *n * *n - 1; idx>= *n * *n - leftover; idx--)
      {
        uo[idx] = uc[idx];
        uc[idx] = un[idx];
      }
    }
    //move the timestamp forward by dt
    (*t) = (*t) + *dt;
}

// simulates the state of the grid after the given time, using a 13-point stencil function
void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */

  double t=0., dt = h / 2.;
  int blocks = (int)pow(n / nthreads, 2);
  int threads = nthreads * nthreads;

  int *blocks_d, *threads_d, *n_d;
  double *un_d, *uc_d, *uo_d, *pebs_d, *t_d, *dt_d, *h_d;

  if (nthreads > n)
  {
    printf("Choose threads less than grid dimension\n");
    return;
  }

  //copy host variables to device
  cudaMalloc( (void **) &un_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &uc_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &uo_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &pebs_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &blocks_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &threads_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &n_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &t_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &dt_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &h_d, sizeof(double) * 1 );

  /* Set up device timers */
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));


  CUDA_CALL(cudaMemcpy( uc_d, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( un_d, u, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( uo_d, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( pebs_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( blocks_d, &blocks, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( threads_d, &threads, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( n_d, &n, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( h_d, &h, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( dt_d, &dt, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( t_d, &t, sizeof(double) * 1, cudaMemcpyHostToDevice ));

  //compute state of the grid over the given time
  while(1)
  {

     evolve<<< blocks, threads >>>(un_d, uc_d, uo_d, pebs_d, n_d, h_d, dt_d, t_d, blocks_d, threads_d);

     //exit from the loop if time exceeds final timestamp
     if(!tpdt(&t,dt,end_time))
        break;

    CUDA_CALL(cudaMemcpy( t_d, &t, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  }
  CUDA_CALL(cudaMemcpy( u, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost ));

  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

  //free resources
  cudaFree(un_d);
  cudaFree(uc_d);
  cudaFree(uo_d);
  cudaFree(blocks_d);
  cudaFree(threads_d);
  cudaFree(pebs_d);
  cudaFree(n_d);
  cudaFree(t_d);
  cudaFree(h_d);
  cudaFree(dt_d);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
