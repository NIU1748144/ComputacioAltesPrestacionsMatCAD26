CC     := gcc
OPT    := -Ofast
LDLIBS := -lm

SRC := main.c \
       common/common.c \
       configuration/config.c \
       layer/layer.c \
       randomizer/randomizer.c \
       initialize/initialize.c \
       training/training.c

CC     := gcc
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

.PHONY: all seq par clean

all: seq par
seq: exec_seq
par: exec_par

exec_seq: $(SRC) $(HDR)
	$(CC) $(OPT) $(SRC) -o $@ $(LDLIBS)

exec_par: $(SRC) $(HDR)
	$(CC) $(OPT) -fopenmp $(SRC) -o $@ $(LDLIBS)

clean:
	rm -f exec_seq exec_par
