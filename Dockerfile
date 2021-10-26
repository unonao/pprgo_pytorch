FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu102.html

RUN apt-get update && \
    apt-get install -y build-essential cmake clang libssl-dev vim git

COPY ./pprgo ./pprgo
COPY ./setup.py .
RUN pip install -e .
RUN pip install jupyterlab jupyter

WORKDIR /home/
