#!/bin/bash
nvcc $FLAGS rainbow.cu -Xptxas -v -O3 -arch=sm_75 -o rainbow

