# Makefile for SU2Gauge.cc on cuth00.phys.columbia.edu

# Compiler
COMPILER = g++

# Flags
CPP_FLAG = 

#OpenMP
OMP_FLAG = -fopenmp

# GSL
GSL_DIR = /usr
GSL_INC = -I$(GSL_DIR)/include
GSL_LIB = -L$(GSL_DIR)/lib -lgsl -lgslcblas

all: SU2Gauge.o statJKS.o
	$(COMPILER) -O3 -o SU2Gauge.out SU2Gauge.o statJKS.o \
	$(GSL_LIB) $(CPP_FLAG) $(OMP_FLAG)
	
SU2Gauge.o: SU2Gauge.cc
	$(COMPILER) -c SU2Gauge.cc \
	$(CFITSIO_INC) $(GSL_INC) $(HEALPIX_INC) $(CPP_FLAG) $(OMP_FLAG)
	
statJKS.o: statJKS.cc
	$(COMPILER) -c statJKS.cc
	
IO: IO.o
	$(COMPILER) -O3 -o IO.out IO.o \
	$(CFITSIO_LIB) $(GSL_LIB) $(HEALPIX_LIB) $(CPP_FLAG) $(OMP_FLAG) $(CFITSIO_LIB)
	
IO.o: IO.cc
	$(COMPILER) -c IO.cc \
	$(CFITSIO_INC) $(GSL_INC) $(HEALPIX_INC) $(CPP_FLAG) $(OMP_FLAG)
	
GDB: SU2Gauge-g.o statJKS-g.o
	$(COMPILER) -gstabs+ -O3 -o SU2Gauge-g.out SU2Gauge-g.o statJKS-g.o \
	$(GSL_LIB) $(CPP_FLAG) $(OMP_FLAG)
        
SU2Gauge-g.o: SU2Gauge.cc
	$(COMPILER) -gstabs+ -c -o SU2Gauge-g.o SU2Gauge.cc \
	$(CFITSIO_INC) $(GSL_INC) $(HEALPIX_INC) $(CPP_FLAG) $(OMP_FLAG)
        
statJKS-g.o: statJKS.cc
	$(COMPILER) -gstabs+ -c -o statJKS-g.o statJKS.cc
