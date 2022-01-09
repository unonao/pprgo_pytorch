#!/bin/sh -l

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g gs54
#PJM -j

module load python/3.8.12
module load cuda/11.1

source venv/bin/activate # ← 仮想環境を activate

pip3 install --upgrade pip setuptools wheel
pip3 install torch==1.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -e .
