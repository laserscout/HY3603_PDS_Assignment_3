######################################################################
 #
 # Makefile -- The makefile for our CUDA app
 #
 # Inspiration from Patrick:
 # https://github.com/hellopatrick/cuda-samples/
 # blob/master/game_of_life_graphics
 #
 # Frank Blanning <frankgou@auth.gr>
 # John Flionis < @auth.gr>
 #
######################################################################

PROJECT_NAME = near

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = gcc

CUDAPATH = /usr/local/cuda

CFLAGS = -c -I $(CUDAPATH)/include
NVCCFLAGS = -c -I $(CUDAPATH)/include -Wno-deprecated-gpu-targets

LFLAGS = -L$(CUDAPATH)/lib -lcuda -lcurand -Wno-deprecated-gpu-targets -ftz=true #here of above?

all: build run

build: gpu #cpu
	$(NVCC) $(LFLAGS) -o $(PROJECT_NAME) *.o

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

# Try and fix this later so that if a .c file doesn't exist, skip
# the call to gcc.
#cpu:			
#	$(CC) $(CFLAGS) *.c

clean:
	rm *.o

run:
	./$(PROJECT_NAME) 5 2 1

