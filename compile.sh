#!/bin/bash

ARCH="sm_75"
FLAGS="-Xptxas -v -O3 -arch=${ARCH}"

if [ "$1" == "1" ]; then
    nvcc $FLAGS rainbow.cu -o rainbow
elif [ "$1" == "2" ]; then
    nvcc $FLAGS rainbow_sliced.cu -o rainbow_sliced
else
    echo "Błąd: Podaj argument 1 lub 2"
    echo "  1 = klasyczna wersja (rainbow.cu)"
    echo "  2 = bitsliced wersja (rainbow_sliced.cu)"
    exit 1
fi

