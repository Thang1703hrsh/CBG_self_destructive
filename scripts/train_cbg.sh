#!/bin/bash
# CBG uses a single-process geometry implementation for the first validated run.

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    --method cbg \
    --config config/cbg_llama32_3b.yaml
