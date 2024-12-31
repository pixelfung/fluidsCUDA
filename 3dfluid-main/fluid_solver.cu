#include "fluid_solver.h"
#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

//define as threads necessarias para processar todos os k's, so precisamos metade das threads nesta dimensao (nos steps do solver)
#define tz 2
//define as threads necessarias para processar todos os j's e i's (nos steps do solver)
#define ty 2
#define tx 32
//define a quantidade de blocos necessaria para processar as 3D (dg,dg,dg)
#define dg 6
//define a quantidade de blocos de forma a ter metade das threads na dim i nos steps do solver
#define dx 3
//define a quantidade de blocos para usar nos steps do solver de forma a preencher as dim j,k
#define dy 84
#define dz 84
//define a quantidade de threads que juntamente com dg preenchem todo o espaco 3D
#define tg 32
//calcula as threads  por bloco dos steps do solver
#define THREADS_PER_BLOCK tz*ty*tx
 //define as threads para a redução em 1 bloco
#define THR_RED 128
//calcula o numero total de blocos
#define numBlocks dx*dy*dz
//define os blocos que são necessários à redução em um bloco
#define AUX ((numBlocks + THR_RED - 1) / THR_RED)
#define CALC (31 - __builtin_clz(AUX))

#define Lblock (1 << (CALC + 1))
//tamanho array max_c
#define C_size numBlocks*AUX
#define SWAP(x0, x)                                                            \
{                                                                            \
float *tmp = x0;                                                           \
x0 = x;                                                                    \
x = tmp;                                                                   \
}


#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

//CUDA KERNELS USADOS NAS DIVERSAS FUNÇÕES:

	//kernel add_source
	__global__ void add_source_kernel(float *x, float *s, float dt) {
	  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	  // Verificando se o índice está dentro dos limites válidos
	  x[i] += dt * s[i];
	}
	//kernel's set_bnd
	__global__ void loop1(float *x,int b, int M, int N, int O) {
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x+1;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y+1;
		if (i <=M && j <=N) {
			x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
			x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
		}
	}
	__global__ void loop2(float *x,int b, int M, int N, int O) {
		unsigned int j = blockIdx.x * blockDim.x + threadIdx.x+1;
		unsigned int k = blockIdx.y * blockDim.y + threadIdx.y+1;
		if (j<=N && k<=O){
			x[IX(0, j, k)] = b == 1 ? -x[IX(1, j, k)] : x[IX(1, j, k)];
			x[IX(M + 1, j, k)] = b == 1 ? -x[IX(M, j, k)] : x[IX(M, j, k)];
		}
	}
	__global__ void loop3(float *x,int b, int M, int N, int O) {
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x+1;
		unsigned int k = blockIdx.y * blockDim.y + threadIdx.y+1;
		if  (i<=M && k<=O){
			x[IX(i, 0, k)] = b == 2 ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
			x[IX(i, N + 1, k)] = b == 2 ? -x[IX(i, N, k)] : x[IX(i, N,k)];
		}
	}
	__global__ void set_corner(float *x, int M, int N) {
		int bid=blockIdx.x;
		if (bid==0)
		x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

		if (bid==1)
		x[IX(M + 1, 0, 0)] =
			0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

		if (bid==2)
		x[IX(0, N + 1, 0)] =
			0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

		if (bid==3)
		x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
										  x[IX(M + 1, N + 1, 1)]);
	}
	//Kernel's lin_solve
		__device__ int IX_sh(int i,int j,int k) {
			return (i>>1)+(j*(tx+1))+(k*(ty+2)*(tx+1));
	}
		//kernel's para passos black/red
		__global__ void lin_solve_step1(int M, int N, int O, float *x, float *x0, float a, float c, float *max_c) {
			__shared__ float shared_max[THREADS_PER_BLOCK];
			__shared__ float sh_x[(tz+2)*(ty+2)*(tx+1)];
			//indices i,j,k i usaod de forma a andar de 2 em 2 sem desperdiçar threads assim so precisamos de metade na dim i7
			unsigned int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
			unsigned int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
			unsigned int i = 2 * (threadIdx.x + blockIdx.x * blockDim.x) + 1 + (k + j) % 2;

			//ids locais
			int lj= threadIdx.y+1;
		    int lk= threadIdx.z+1;
			int li= 2*threadIdx.x+1+(lj+lk)%2;

			unsigned int tid = threadIdx.x + threadIdx.y * tx + threadIdx.z * (ty * tx);
			unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);

			if (i<=M+2 && j<=N+2 && k<=O+2) {
				//adicionar a shared
				sh_x[IX_sh(li-1,lj,lk)]=x[IX(i-1,j,k)]; //adicionado a verde

				if (lk==1)sh_x[IX_sh(li,lj,lk-1)]=x[IX(i,j,k-1)]; //adicionado a vermelho
				if (lj==1) sh_x[IX_sh(li,lj-1,lk)]=x[IX(i,j-1,k)]; //adicionado a roxo
				if (li==2*tx || li==2*tx-1){
					if (i+1<(M+2))
						sh_x[IX_sh(li+1,lj,lk)]=x[IX(i+1,j,k)]; //adicionado a laranja
					else sh_x[IX_sh(li+1,lj,lk)]=0.0f;
				}

				if (lj==ty){
					if (j+1<N+2)
						sh_x[IX_sh(li,lj+1,lk)]=x[IX(i,j+1,k)]; //adicionado a branco
					else sh_x[IX_sh(li,lj+1,lk)]=0.0f;
				}
				if (lk==tz) {
					if (k+1<O+2)
						sh_x[IX_sh(li,lj,lk+1)]=x[IX(i,j,k+1)];//adicionado a azul escuro
					else sh_x[IX_sh(li,lj,lk+1)]=0.0f;
				}
			}
			__syncthreads();
			if (i<=M && j<=N && k<=O){
				float x_0=x0[IX(i,j,k)];float old_x = x[IX(i, j, k)];
		  		//calcula x(i,j,k) e armazena o change num array para posteriror redução
		  		float novo=(x_0 + a * (sh_x[IX_sh(li - 1, lj, lk)] + sh_x[IX_sh(li + 1, lj, lk)] +
										   sh_x[IX_sh(li, lj - 1, lk)] + sh_x[IX_sh(li, lj + 1, lk)] +
										   sh_x[IX_sh(li, lj, lk - 1)] + sh_x[IX_sh(li, lj, lk + 1)])) / c;
				x[IX(i, j, k)] = novo;
				shared_max[tid]= fabs(novo - old_x);
			}
			else shared_max[tid]=0.0f;
			__syncthreads();

			// Reduction to find the maximum change in shared memory
			for (int stride = THREADS_PER_BLOCK/2; stride > 0; stride>>=1) {
				if (tid < stride){
				shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
				}
				__syncthreads();
			}

			if (tid == 0) {
				max_c[bid]=shared_max[0];
			  }

		}
		__global__ void lin_solve_step2(int M, int N, int O, float *x, float *x0, float a, float c, float *max_c) {
			__shared__ float shared_max[THREADS_PER_BLOCK];

			__shared__ float sh_x[(tz+2)*(ty+2)*(tx+1)];

			unsigned int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
			unsigned int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
			unsigned int i = 2 * (threadIdx.x + blockIdx.x * blockDim.x) + 1 + (k + j+1) % 2;

			//ids locais

			int lj= threadIdx.y+1;
			int lk= threadIdx.z+1;
			int li= 2*threadIdx.x+1+(lj+lk+1)%2;

			unsigned int tid = threadIdx.x + threadIdx.y * tx + threadIdx.z * (ty * tx);
			unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
			if (i<=M+2 && j<=N+2 && k<=O+2) {
				//adicionar a shared
				sh_x[IX_sh(li-1,lj,lk)]=x[IX(i-1,j,k)]; //adicionado a verde

				if (lk==1)sh_x[IX_sh(li,lj,lk-1)]=x[IX(i,j,k-1)]; //adicionado a vermelho
				if (lj==1) sh_x[IX_sh(li,lj-1,lk)]=x[IX(i,j-1,k)]; //adicionado a roxo
				if (li==2*tx || li==2*tx-1){
					if (i+1<(M+2))
						sh_x[IX_sh(li+1,lj,lk)]=x[IX(i+1,j,k)]; //adicionado a laranja
					else sh_x[IX_sh(li+1,lj,lk)]=0.0f;
				}

				if (lj==ty){
					if (j+1<N+2)
						sh_x[IX_sh(li,lj+1,lk)]=x[IX(i,j+1,k)]; //adicionado a branco
					else sh_x[IX_sh(li,lj+1,lk)]=0.0f;
				}
				if (lk==tz) {
					if (k+1<O+2)
						sh_x[IX_sh(li,lj,lk+1)]=x[IX(i,j,k+1)];//adicionado a azul escuro
					else sh_x[IX_sh(li,lj,lk+1)]=0.0f;
				}
			}
			__syncthreads();
			if (i<=N && j<=M && k<=O){
				//calcula x(i,j,k) e armazena o change num array para posteriror redução
				//calcula x(i,j,k) e armazena o change num array para posteriror redução
				float old_x = x[IX(i, j, k)];
				float novo=(x0[IX(i, j, k)] + a * (sh_x[IX_sh(li - 1, lj, lk)] + sh_x[IX_sh(li + 1, lj, lk)] +
										 sh_x[IX_sh(li, lj - 1, lk)] + sh_x[IX_sh(li, lj + 1, lk)] +
										 sh_x[IX_sh(li, lj, lk - 1)] + sh_x[IX_sh(li, lj, lk + 1)])) / c;

				x[IX(i, j, k)] = novo;
				shared_max[tid]= fabs(novo - old_x);
			}
			else shared_max[tid]=0.0f;
			__syncthreads();

			// Reduction to find the maximum change in shared memory
			for (int stride = THREADS_PER_BLOCK/2; stride > 0; stride >>= 1) {
				if (tid < stride) {
					shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
				}
				__syncthreads();
			}

			if (tid == 0) {
				max_c[bid]=fmaxf(shared_max[0],max_c[bid]);
			}
		}
		//kernel's redução
		__global__ void reduce_to_block(float *blocks_max, float *global_maxs) {
			//alocamos um array que tenha uma posição para cada thread dentro do bloco
			__shared__ float shared_max[THR_RED];

			// definimos os ids locais e globais
			int tid = threadIdx.x;
			int idx = threadIdx.x + blockIdx.x * blockDim.x*2;

			//alocamos a memoria da global para a shared
			if (idx+THR_RED<numBlocks) shared_max[tid] = fmaxf(blocks_max[idx],blocks_max[idx+blockDim.x]);
                        else if (idx<numBlocks) shared_max[tid] = blocks_max[tid];                      
                        else shared_max[tid]=0.0f;
			__syncthreads();

			// Redução paralela para encontrar o máximo no intervalo de indices do bloco
			for (int stride = THR_RED/2; stride > 0; stride >>= 1) {
				if (tid < stride) {
					shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
				}
				__syncthreads();
			}
			// O primeiro thread do bloco armazena o máximo final na memória global
			if (tid == 0) {
				global_maxs[blockIdx.x]=shared_max[0];
			}
			//obtemos assim um array mais pequeno que pode ser reduzido a um unico float em outro kernel
		}
		__global__ void reduce_max(float *block_max, float *global_max) {
			//aloca array shared com posiçoes suficientes ao array pequeno ja reduzido
			__shared__ float shared_max[Lblock/2];
			//id local
		  int tid = threadIdx.x;

		  // Inicializa a memória compartilhada
		  shared_max[tid] = block_max[tid];

		  __syncthreads();

		  // Redução paralela para encontrar o máximo global
		  for (int stride = Lblock/4; stride > 0; stride >>=1) {
		    if (tid < stride) {
		      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
		    }
		    __syncthreads();
		  }

		  // O primeiro thread do bloco armazena o máximo final na memória global
		  if (tid == 0) {
		    *global_max=shared_max[0];
		  }
		}
	//Kernel advect
	__global__ void advect_kernel(int M, int N, int O, float *d, float *d0, float *u, float *v, float *w, float dt) {
	  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
	  if (i <= M && j <= N && k <= O) {
	    float x = i - dtX * u[IX(i, j, k)];
	    float y = j - dtY * v[IX(i, j, k)];
	    float z = k - dtZ * w[IX(i, j, k)];

	    // Clamp to grid boundaries
	    x = (x < 0.5f) ? 0.5f : (x > M + 0.5f) ? M + 0.5f : x;
	    y = (y < 0.5f) ? 0.5f : (y > N + 0.5f) ? N + 0.5f : y;
	    z = (z < 0.5f) ? 0.5f : (z > O + 0.5f) ? O + 0.5f : z;

	    int i0 = (int)x, i1 = i0 + 1;
	    int j0 = (int)y, j1 = j0 + 1;
	    int k0 = (int)z, k1 = k0 + 1;

	    float s1 = x - i0, s0 = 1 - s1;
	    float t1 = y - j0, t0 = 1 - t1;
	    float u1 = z - k0, u0 = 1 - u1;

	    d[IX(i, j, k)] =
	        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
	              t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
	        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
	              t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
	  }
	}
	//Kernel's project
	__global__ void project1(float *u, float *v, float *w, float *div, float *p,int M, int N, int O) {
		  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x+1;
		  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y+1;
		  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z+1;

		  // Verificando se o índice está dentro dos limites válidos da grade
		  if (i <= M && j <= N && k <= O) {
		    // Calculando div conforme o código fornecido
		    div[IX(i, j, k)] =
		        -0.5f * (
		            u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
		            v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
		            w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
		        ) / fmaxf(M, fmaxf(N, O));

		    // Inicializando p
		    p[IX(i, j, k)] = 0.0f;
		  }
		}
	__global__ void project2(float *u, float *v, float *w, float *p, int M, int N, int O) {
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x+1;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y+1;
		unsigned int k = blockIdx.z * blockDim.z + threadIdx.z+1;

		// Verificando se o índice está dentro dos limites válidos da grade
		if ( i <= M &&  j <= N  && k <= O) {
		  // Atualizando u
		  u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);

		  // Atualizando v
		  v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);

		  // Atualizando w
		  w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
		}
	}












//FUNÇÕES DE CPU
	// Add sources (density or velocity)
	void add_source(int M, int N, int O, float *x, float *s, float dt) {
	  int size = (M + 2) * (N + 2) * (O + 2);
	  add_source_kernel<<<(int) size/1000, 1000>>>(x, s, dt);
	}

	// Set boundary conditions
	void set_bnd(int M, int N, int O, int b, float *x) {
		// Set boundary on faces
		loop1<<<dim3 (dg,dy,1),dim3 (tg,ty,1)>>>(x,b,M,N,O);

		loop2<<<dim3 (dg,dy,1),dim3 (tg,ty,1)>>>(x,b,M,N,O);

		loop3<<<dim3 (dg,dy,1),dim3 (tg,ty,1)>>>(x,b,M,N,O);
		// Set corners
		set_corner<<<4,1>>>(x,M,N);
	}

	// Linear solve for implicit methods (diffusion)
	// red-black solver with convergence check
	void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
	  float h_max_c;

	  dim3 threads(tx,ty,tz); // Ajustar com base no problema
	  dim3 blocks(dx, dy, dz);

	  float tol = 1e-7;
	  int l = 0;

	  do {

	    // Configuração do grid e blocos

	    lin_solve_step1<<<blocks, threads>>>(M, N, O, x, x0, a, c, aux_max_c);
  		//CUDA_CHECK_ERROR();
	    lin_solve_step2<<<blocks, threads>>>(M, N, O, x, x0, a, c, aux_max_c);
  		//CUDA_CHECK_ERROR();
	    reduce_to_block<<<Lblock/2,THR_RED>>>(aux_max_c, block_max_c);
	    reduce_max<<<1, Lblock/2>>>(block_max_c, d_max_c);

	    set_bnd(M, N, O, b, x); // Chamar função `set_bnd
	  	cudaMemcpy(&h_max_c,d_max_c, sizeof(float), cudaMemcpyDeviceToHost);
	    //printf("iteration while: %d\n",l);
	  } while (h_max_c > tol && ++l < 20);
	}

	// Diffusion step (uses implicit method)
	void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,float dt) {
	    int max = MAX(MAX(M, N), O);
	    float a = dt * diff * max * max;
	    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
	}

	// Advection step (uses velocity field to move quantities)
	void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
	  dim3 threadsPerBlock(tg, ty, tz);
	  dim3 numblocks(dg,dy,dz);

	  advect_kernel<<<numblocks, threadsPerBlock>>>(M, N, O, d, d0, u, v, w, dt);

	  set_bnd(M, N, O, b, d);
	}

	// Projection step to ensure incompressibility (make the velocity field
	// divergence-free)
	void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
	  dim3 threadsPerBlock(tg, ty, tz);
	  dim3 numblocks(dg,dy,dz);

	  project1<<<numblocks,threadsPerBlock>>>(u, v, w, div, p, M, N, O);

	  set_bnd(M, N, O, 0, div);
	  set_bnd(M, N, O, 0, p);

	  lin_solve(M, N, O, 0, p, div, 1, 6);

	  project2<<<numblocks,threadsPerBlock>>>(u, v, w, p, M, N, O);

	  set_bnd(M, N, O, 1, u);
	  set_bnd(M, N, O, 2, v);
	  set_bnd(M, N, O, 3, w);
	}

	// Step function for density
	void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,float *w, float diff, float dt) {
	  add_source(M, N, O, x, x0, dt);

	  SWAP(x0,x);

	  diffuse(M, N, O, 0, x, x0, diff, dt);

	  SWAP(x0,x);

	  advect(M, N, O, 0, x, x0, u, v, w, dt);
	}

	// Step function for velocity
	void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
	  add_source(M, N, O, u, u0, dt);
	  add_source(M, N, O, v, v0, dt);
	  add_source(M, N, O, w, w0, dt);

	  SWAP(u0,u);

	  diffuse(M, N, O, 1, u, u0, visc, dt);

	  SWAP(v0,v);

	  diffuse(M, N, O, 2, v, v0, visc, dt);

	  SWAP(w0,w);

	  diffuse(M, N, O, 3, w, w0, visc, dt);
	  project(M, N, O, u, v, w, u0, v0);

	  SWAP(u0,u);
	  SWAP(v0,v);
	  SWAP(w0,w);

	  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
	  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
	  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
	  project(M, N, O, u, v, w, u0, v0);
	}
