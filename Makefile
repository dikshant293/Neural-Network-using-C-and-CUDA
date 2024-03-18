compile:
	nvcc nn.cu -use_fast_math -Xcompiler -fopenmp -lopenblas -lcublas -O3 -arch=sm_70 -o nn

plot:
	@chmod 700 plt.sh
	./plt.sh

cpu_blas: compile
	./nn 1 800 50 200 0.1 0 0 0
	@chmod 700 plt.sh
	./plt.sh

cpu_noblas: compile
	./nn 1 800 50 200 0.1 1 0 0
	@chmod 700 plt.sh
	./plt.sh

gpu_blas: compile
	./nn 1 800 50 200 0.1 0 1 0
	@chmod 700 plt.sh
	./plt.sh

gpu_noblas: compile
	./nn 1 800 50 200 0.1 1 1 0
	@chmod 700 plt.sh
	./plt.sh

bench: compile
	@echo "################################## Benchmark ##################################"
	@echo GPU native
	./nn 1 800 50 200 0.1 1 1 0 0
	@echo GPU CuBLAS
	./nn 1 800 50 200 0.1 0 1 0 0
	@echo CPU BLAS
	./nn 1 800 50 200 0.1 0 0 0 0
	@echo CPU native
	./nn 1 800 50 200 0.1 1 0 0 0

random: compile
	./nn 1 800 50 200 0.1 0 1 0 0 1
	./nn 1 800 50 200 0.1 0 1 0 0 2
	./nn 1 800 50 200 0.1 0 1 0 0 4
	./nn 1 800 50 200 0.1 0 1 0 0 8
	./nn 1 800 50 200 0.1 0 1 0 0 16
	./nn 1 800 50 200 0.1 0 1 0 0 32
	./nn 1 800 50 200 0.1 0 1 0 0 64
	./nn 1 800 50 200 0.1 0 1 0 0 128
	./nn 1 800 50 200 0.1 0 1 0 0 256
	./nn 1 800 50 200 0.1 0 1 0 0 512
	./nn 1 800 50 200 0.1 0 1 0 0 1024
	./nn 1 800 50 200 0.1 0 1 0 0 2048

plots: cpu_blas cpu_noblas gpu_blas gpu_noblas
	@echo DONE

clean:
	\rm -f *.o nn nn.exe nn*~ *# 

extract:
	gunzip -k *.gz
