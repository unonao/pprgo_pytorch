#!/bin/sh -l

#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -g gs54
#PJM -j

module load gcc/8.3.1
module load python/3.8.12
module load cuda/11.1
source venv/bin/activate # ← 仮想環境を activate


python3 run.py --config config/sapprgo_pubmed_logspace_nsplit01.yaml
