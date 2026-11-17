CC     := gcc
MPICC  := mpicc
OPT    := -Ofast
LDLIBS := -lm

SRC := main.c \
       common/common.c \
       configuration/config.c \
       layer/layer.c \
       randomizer/randomizer.c \
       initialize/initialize.c \
       training/training.c

HDR := $(wildcard *.h */*.h)

.PHONY: all seq par mpi clean

all: seq par
seq: exec_seq
par: exec_par
mpi: exec_mpi

exec_seq: $(SRC) $(HDR)
	$(CC) $(OPT) $(SRC) -o $@ $(LDLIBS)

exec_par: $(SRC) $(HDR)
	$(CC) $(OPT) -fopenmp $(SRC) -o $@ $(LDLIBS)

exec_mpi: $(SRC) $(HDR)
	$(MPICC) $(OPT) $(SRC) -o $@ $(LDLIBS)
clean:
	rm -f exec_seq exec_par exec_mpi
