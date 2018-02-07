FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    git \
	python-pil \
    python-scipy \
    python-numpy \
    ffmpeg \
    wget

# Download neural-style
RUN git clone https://github.com/lengstrom/fast-style-transfer.git fast-neural-style

# Setup project
RUN cd fast-neural-style; bash setup.sh

# create volume for the images
RUN mkdir /images
VOLUME /images

# Prepare execution environment
WORKDIR /notebooks/fast-neural-style/

CMD python style.py


# docker build -t ahbrosha/fast-neural-style-tf .
# docker run --runtime=nvidia --rm -v $(pwd):/images ahbrosha/fast-neural-style-tf python style.py --style /images/van_gogh_cafe_600.jpg --checkpoint-dir /images --checkpoint-iterations 100
# docker run --runtime=nvidia --rm -v $(pwd):/images ahbrosha/fast-neural-style-tf python evaluate.py --checkpoint /images/test/hodor.ckpt --in-path /images/test/brad_pitt.jpg --out-path /images/test/
