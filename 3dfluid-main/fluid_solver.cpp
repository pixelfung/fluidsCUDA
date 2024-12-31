#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= N; j++) {
      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
    }
  }
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,
               float c) {
  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    for (int i = 1; i <= M; i++) {
      for (int j = 1; j <= N; j++) {
        for (int k = 1; k <= O; k++) {
          x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                            a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                           c;
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

//Total density after 100 timesteps: 81981.3

 //Performance counter stats for './fluid_sim':

   //  2,321,706,151      L1-dcache-load-misses:u
   // 92,345,212,866      instructions:u            #    1.72  insn per cycle
   // 53,627,724,017      cycles:u

     // 17.046181788 seconds time elapsed

      //17.044840000 seconds user
       //0.002000000 seconds sys
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,float c) {
  int k_neig=M*N+2*M+2*N+4;
  float aux1,aux2,aux3;
  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    for (int i = 1;  i <= M; i++) {
      for (int j = 1; j <= N; j++) {
        for (int k = 1; k <= O; k+=3) {
          int idx = IX(i, j, k);
          int idx1 = idx+k_neig;
          int idx2 = IX(i,j,k+2);
          aux1  = (x0[idx]  + a*(x[idx+1] + x[idx-1] + x[idx-M-2]+x[idx+M+2] + x[idx+k_neig] + x[idx-k_neig]))/c;
          aux2 = (x0[idx1] + a*(x[idx1+1] + x[idx1-1] + x[idx1-M-2] + x[idx1+M+2]+x[idx1+k_neig] + aux1))/c;
          aux3 = (x0[idx2] + a*(x[idx2+1] + x[idx2-1] + x[idx2-M-2] + x[idx2+M+2]+ x[idx2+k_neig] + aux2))/c;
          x[idx] = aux1;
          x[idx1] = aux2;
          x[idx2] = aux3;
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

#define BLOCK_SIZE 9
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,float c) {
  int k_neig=(M+2)*(N+2);
  float vare = 1/c;
  for (int l = 0; l < LINEARSOLVERTIMES; l++){
    for(int kk = 1; kk <= O; kk += BLOCK_SIZE){
      for(int jj = 1; jj<= N; jj += BLOCK_SIZE){
        for(int ii = 1; ii <= M; ii += BLOCK_SIZE){
          float aux1,aux2,aux3;
          for (int k = kk; k<= kk + BLOCK_SIZE && k <= O; k+=3) {
            for (int j = jj; j<= jj + BLOCK_SIZE && j <= N; j++) {
              for (int i = ii; i<= ii + BLOCK_SIZE && i <= M; i++) {
                int idx = IX(i, j, k);
                int idx1 = idx+k_neig;
                int idx2 = IX(i,j,k+2);
                aux1  = vare*(x0[idx]  + a*(x[idx+1] + x[idx-1] + x[idx-M-2]+x[idx+M+2] + x[idx+k_neig] + x[idx-k_neig]));
                aux2 = vare*(x0[idx1] + a*(x[idx1+1] + x[idx1-1] + x[idx1-M-2] + x[idx1+M+2]+x[idx1+k_neig] + aux1));
                aux3 = vare*(x0[idx2] + a*(x[idx2+1] + x[idx2-1] + x[idx2-M-2] + x[idx2+M+2]+ x[idx2+k_neig] + aux2));
                x[idx] = aux1;
                x[idx1] = aux2;
                x[idx2] = aux3;
              }
            }
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

#define BLOCK_SIZE 9  // Defina o tamanho do bloco
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
  int k_neig=M*N+2*M+2*N+4;
  float aux1, aux2, aux3; 
  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    // Iterate over blocks in the i dimension
    for (int kk = 1; kk <= O; kk += BLOCK_SIZE) {  // AQUI TROQUEI
      // Iterate over blocks in the j dimensio
      for (int jj = 1; jj <= N; jj += BLOCK_SIZE) {
        // Iterate over blocks in the k dimension
        for (int ii = 1; ii <= M; ii += BLOCK_SIZE) { // COM ISTO
          // Process each block
          for (int k = kk; k < kk + BLOCK_SIZE && k <= O; k += 3) { // AGORA FOI AQUI
            for (int j = jj; j < jj + BLOCK_SIZE && j <= N; j++) {
               for (int i = ii; i < ii + BLOCK_SIZE && i <= M; i++){ // COM ISTO
                int idx = IX(i, j, k);
                int idx1 = idx+k_neig;
                int idx2 = IX(i,j,k+2);
                aux1  = (x0[idx]  + a*(x[idx+1] + x[idx-1] + x[idx-M-2]+x[idx+M+2] + x[idx+k_neig] + x[idx-k_neig]))/c;
                aux2 = (x0[idx1] + a*(x[idx1+1] + x[idx1-1] + x[idx1-M-2] + x[idx1+M+2]+x[idx1+k_neig] + aux1))/c;
                aux3 = (x0[idx2] + a*(x[idx2+1] + x[idx2-1] + x[idx2-M-2] + x[idx2+M+2]+ x[idx2+k_neig] + aux2))/c;
                x[idx] = aux1;
                x[idx1] = aux2; 
                x[idx2] = aux3; 
              }
            }
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

// Linear solve for implicit methods (diffusion)
#define BLOCK_SIZE 9  // Defina o tamanho do bloco
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
  int k_neig=M*N+2*M+2*N+4;
  float vare = a/c;
  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    // Iterate over blocks in the i dimension
    for (int kk = 1; kk <= O; kk += BLOCK_SIZE) {  // AQUI TROQUEI
      // Iterate over blocks in the j dimension
      for (int jj = 1; jj <= N; jj += BLOCK_SIZE) {
        // Iterate over blocks in the k dimension
        for (int ii = 1; ii <= M; ii += BLOCK_SIZE) { // COM ISTO
          // Process each block
          for (int k = kk; k < kk + BLOCK_SIZE && k <= O; k += 3) { // AGORA FOI AQUI
            for (int j = jj; j < jj + BLOCK_SIZE && j <= N; j++) {
               for (int i = ii; i < ii + BLOCK_SIZE && i <= M; i++){ // COM ISTO
                int idx = IX(i, j, k);
                int idx1 = idx+k_neig;
                int idx2 = IX(i,j,k+2);
                x[idx]  = vare*(x0[idx]/a  + (x[idx+1] + x[idx-1] + x[idx-M-2]+x[idx+M+2] + x[idx+k_neig] + x[idx-k_neig]));
                x[idx1] = vare*(x0[idx1]/a + (x[idx1+1] + x[idx1-1] + x[idx1-M-2] + x[idx1+M+2]+x[idx1+k_neig] + x[idx1-k_neig]));
                x[idx2] = vare*(x0[idx2]/a + (x[idx2+1] + x[idx2-1] + x[idx2-M-2] + x[idx2+M+2]+ x[idx2+k_neig] + x[idx2-k_neig]));
              }
            }
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

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
  }
  set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {
  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        div[IX(i, j, k)] =
            -0.5f *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
            MAX(M, MAX(N, O));
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
