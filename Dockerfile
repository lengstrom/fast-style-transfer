FROM tensorflow/tensorflow:0.11.0

# install packages
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    git wget && \
    apt-get autoclean && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# install python moviepy
ENV PYTHON_VERSION 2.7
RUN pip2 install \
    numpy==1.11.2 \
    Pillow==3.4.2 \
    scipy==0.18.1 \
    moviepy==1.0.2

# setup working directory
WORKDIR /app

# download pre-trained data
RUN curl https://1drv.ms/u/s!AlkXAYHD4YTwl8smmon3XYw_qhDWTQ?e=TP9DBc --output la_muse.ckpt && \
    curl https://1drv.ms/u/s!AlkXAYHD4YTwl8soAzsbmBRj0l7vIg?e=ngkFzZ --output rain_princess.ckpt && \
    curl https://1drv.ms/u/s!AlkXAYHD4YTwl8sp8SAp05DrAW71UQ?e=TKWvqv --output scream.ckpt && \
    curl https://1drv.ms/u/s!AlkXAYHD4YTwl8smmon3XYw_qhDWTQ?e=9lO9cX --output udnie.ckpt && \
    curl https://1drv.ms/u/s!AlkXAYHD4YTwl8slmjVb78BeY2SyzA?e=k0d5aE --output wave.ckpt && \
    curl https://1drv.ms/u/s!AlkXAYHD4YTwl8sn_3DKGXARqlzYtg?e=ERZJZF --output wreck.ckpt

# clone the repository and setup python path
RUN git clone https://github.com/thhuang/fast-style-transfer.git && \
    cd fast-style-transfer && \
    git reset --hard 57d473706d92759a68543f6392bb7cfcb0a9a35b
ENV PYTHONPATH /app/fast-style-transfer/src

# default command
CMD python fast-style-transfer/evaluate.py --checkpoint input/udnie.ckpt --in-path input/stata.jpg --out-path output/stata-udnie.png
