CPP = nvcc -O3 -Wno-deprecated-gpu-targets -maxrregcount=32 -use_fast_math
#CPP=mpicxx -Wall -Ofast -ftree-vectorize -funroll-loops -fopenmp
#CPP=g++ -Wall -Ofast -ftree-vectorize -funroll-loops -fopenmp
SRCS = main.cu fluid_solver.cu EventManager.cpp
#SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

runseq:
	./fluid_sim_seq
	
run:
	mpirun -np 2