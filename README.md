# 3D Fluid Simulation Project

## Overview
This project involves the simulation of fluid dynamics in a three-dimensional grid. It leverages numerical methods, such as **advection** and **diffusion**, to model fluid behavior over time. The focus is on optimizing the performance of critical functions like `lin_solve`, which handles linear equation solving, by improving locality, vectorization, and spatial efficiency.

## Features
- **3D Grid-Based Fluid Simulation**: Models fluid flow in a discretized 3D space.
- **Efficient Numerical Methods**: Implements advection, diffusion, and linear solvers.
- **Optimization-Oriented Design**: Focuses on reducing execution time through algorithmic improvements without relying on external libraries.

## Requirements
To run the project, you need:

- A C compiler (e.g., GCC or Clang)
- CUDA Toolkit (optional, for GPU acceleration)
- Git (for version control, if cloning from GitHub)

### Optional Tools
- Python (if using post-processing or visualization scripts)
- Visualization tools like ParaView or matplotlib (for analyzing output data)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fluidsCUDA.git
   cd fluidsCUDA
   ```

2. Build the project:
   ```bash
   make
   ```

3. Run the simulation:
   ```bash
   ./simulate
   ```


## Key Components
- **`lin_solve`**: Solves linear equations iteratively. Optimization efforts focus on improving its spatial and temporal locality.
- **Advection and Diffusion**: Core components for modeling fluid dynamics.
- **Boundary Conditions**: Ensures realistic simulation behavior at the edges of the grid.

## Optimization Strategies
- **Locality**: Minimizing cache misses by restructuring data access patterns.
- **Vectorization**: Leveraging SIMD instructions for faster computation.
- **Spatial Efficiency**: Optimizing memory usage to handle large 3D grids.

## How to Contribute
Contributions are welcome! Here’s how you can help:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork.
4. Open a pull request.

## Acknowledgments
This project is part of ongoing research and experimentation in fluid dynamics simulation. Special thanks to the contributors and the open-source community for providing inspiration and support.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
