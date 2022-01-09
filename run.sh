#!/bin/sh -l

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:15:00
#PJM -g gs54
#PJM -j

module load python/3.8.12
module load cuda/11.1
source venv/bin/activate # ← 仮想環境を activate

python3 demo.py
