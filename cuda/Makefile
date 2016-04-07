NVCC = nvcc
DEPS = diff.h
CFLAGS=-I.

#change arch and code according to your hardware
CUDA_OPTIONS = -ptx -arch=compute_30 -code=sm_30 --ptxas-options=-v

all: tvq2opt tvxopt

sep: tvq2opt_sep tvxopt_sep

tvq2opt: tvq2opt.cu $(DEPS)
	$(NVCC) $(CUDA_OPTIONS) -o $@.ptx $< $(CFLAGS) 1>&2

tvxopt: tvxopt.cu $(DEPS)
	$(NVCC) $(CUDA_OPTIONS) -o $@.ptx $< $(CFLAGS) 1>&2

tvq2opt_sep: tvq2opt_sep.cu $(DEPS)
	$(NVCC) $(CUDA_OPTIONS) -o $@.ptx $< $(CFLAGS) 1>&2
	
tvxopt_sep: tvxopt_sep.cu $(DEPS)
	$(NVCC) $(CUDA_OPTIONS) -o $@.ptx $< $(CFLAGS) 1>&2

clean:
	rm *.ptx
