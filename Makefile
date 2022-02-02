NVCC=nvcc

.PHONY: fmt
fmt:
	clang-format -i *.{cu,cuh}

.PHONY: shaminer
shaminer:
	$(NVCC) -o shaminer shaminer.cu
