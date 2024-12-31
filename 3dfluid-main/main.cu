#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda.h>
#define SIZE 168

//define a quantidade de blocos de forma a ter metade das threads na dim i nos steps do solver
#define dx 3
//define a quantidade de blocos para usar nos steps do solver de forma a preencher as dim j,k
#define dy 84
#define dz 84
 //define as threads para a redução em 1 bloco
#define THR_RED 128
//calcula o numero total de blocos
#define numBlocks dx*dy*dz
//define os blocos que são necessários à redução em um bloco
#define AUX ((numBlocks + THR_RED - 1) / THR_RED)


#define Lblock (256)
//tamanho array max_c
#define C_size numBlocks*AUX
#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *dens,*u,*v,*w;
float *du,*du_prev,*dv,*dv_prev,*dw,*dw_prev,*ddens,*ddens_prev;
float *d_max_c, *aux_max_c,*block_max_c;

int allocate_data_gpu() {
  int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float); // Tamanho em bytes

  // Alocando memória na GPU para cada array
  cudaError_t  err = cudaMalloc((void**)&du, size);
  cudaError_t err_v = cudaMalloc((void**)&dv, size);
  cudaError_t err_w = cudaMalloc((void**)&dw, size);
  cudaError_t err_u_prev = cudaMalloc((void**)&du_prev, size);
  cudaError_t err_v_prev = cudaMalloc((void**)&dv_prev, size);
  cudaError_t err_w_prev = cudaMalloc((void**)&dw_prev, size);
  cudaError_t err_dens = cudaMalloc((void**)&ddens, size);
  cudaError_t err_dens_prev = cudaMalloc((void**)&ddens_prev, size);
  cudaMalloc((void **)&d_max_c, sizeof(float));
  cudaMalloc((void **)&aux_max_c, sizeof(float)*C_size);
  cudaMalloc((void **)&block_max_c, sizeof(float)*Lblock);
  // Verificando se algum erro ocorreu
  if (err != cudaSuccess || err_v != cudaSuccess || err_w != cudaSuccess ||
      err_u_prev != cudaSuccess || err_v_prev != cudaSuccess ||
      err_w_prev != cudaSuccess || err_dens != cudaSuccess ||
      err_dens_prev != cudaSuccess) {

    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    return 0;
      }

  // Se todas as alocações forem bem-sucedidas
  return 1;
}
// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  dens = new float[size];
  u = new float[size];
  v = new float[size];
  w = new float[size];
  if (!dens || !u || !v || !w) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}
// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    dens[i] =u[i]=v[i]=w[i]= 0.0f;
  }
}
void clear_data_gpu() {
  int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float); // Tamanho em bytes

  // Definindo os valores das variáveis para 0.0f na GPU
  cudaError_t err;

  // Tentando limpar todas as variáveis com cudaMemset
  err = cudaMemset(du, 0, size);
  cudaError_t err_v = cudaMemset(dv, 0, size);
  cudaError_t err_w = cudaMemset(dw, 0, size);
  cudaError_t err_u_prev = cudaMemset(du_prev, 0, size);
  cudaError_t err_v_prev = cudaMemset(dv_prev, 0, size);
  cudaError_t err_w_prev = cudaMemset(dw_prev, 0, size);
  cudaError_t err_dens = cudaMemset(ddens, 0, size);
  cudaError_t err_dens_prev = cudaMemset(ddens_prev, 0, size);

  // Verificando se algum erro ocorreu após todas as chamadas de cudaMemset
  if (err != cudaSuccess || err_v != cudaSuccess || err_w != cudaSuccess ||
      err_u_prev != cudaSuccess || err_v_prev != cudaSuccess ||
      err_w_prev != cudaSuccess || err_dens != cudaSuccess ||
      err_dens_prev != cudaSuccess) {

    std::cerr << "CUDA Memset failed: " << cudaGetErrorString(err) << std::endl;
      }
}

// Free allocated memory
void free_data() {
  delete[] dens;
  delete[] u;
  delete[] v;
  delete[] w;
}
void free_data_gpu() {
  cudaFree(du);
  cudaFree(dv);
  cudaFree(dw);
  cudaFree(du_prev);
  cudaFree(dv_prev);
  cudaFree(dw_prev);
  cudaFree(ddens);
  cudaFree(ddens_prev);
  cudaFree(aux_max_c);
  cudaFree(d_max_c);
  cudaFree(block_max_c);
}
// Apply events (source or force) for the current timestep

void apply_events(const std::vector<Event> &events) {
  for (const auto &event : events) {
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      int i = M / 2, j = N / 2, k = O / 2;
      float density = event.density;
      cudaMemcpy(&ddens[IX(i, j, k)], &density, sizeof(float), cudaMemcpyHostToDevice);
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      int i = M / 2, j = N / 2, k = O / 2;
      float fx = event.force.x, fy = event.force.y, fz = event.force.z;

      cudaMemcpy(&du[IX(i, j, k)], &fx, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&dv[IX(i, j, k)], &fy, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&dw[IX(i, j, k)], &fz, sizeof(float), cudaMemcpyHostToDevice);
    }
  }
}

// Function to sum the total density
double sum_density() {
  double total_density = 0.0;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);
    // Perform the simulation steps
    vel_step(M, N, O, du, dv, dw, du_prev, dv_prev, dw_prev, visc, dt);
    dens_step(M, N, O, ddens, ddens_prev, du, dv, dw, diff, dt);

  }
  cudaMemcpy(dens,ddens,size,cudaMemcpyDeviceToHost);
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  //alocar memoria na GPU
  allocate_data_gpu();
  if (!allocate_data())
    return -1;
  clear_data();
  clear_data_gpu();
  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  double total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();
  //libera a memoria alocada na GPU
  free_data_gpu();
  return 0;
}