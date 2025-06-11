all:
	nvcc code/rainbow.cu -Xptxas -v -O3 -arch=sm_75 -o rainbow

clean:
	rm -f rainbow

