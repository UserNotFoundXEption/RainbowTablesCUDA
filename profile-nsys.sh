#!/bin/bash
sudo nsys profile -t cuda,nvtx,osrt --stats=true --gpu-metrics-devices=all ./run.sh
