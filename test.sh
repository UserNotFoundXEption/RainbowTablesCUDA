sudo nsys profile -t cuda,nvtx,osrt --stats=true --gpu-metrics-devices=all ./run.sh
head rainbow_des.txt -n 3
tail rainbow_des.txt -n 3
