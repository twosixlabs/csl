FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Install Conda
RUN apt-get update --fix-missing && \
    apt-get install -y wget curl git vim build-essential
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo 'alias ll="ls -al"' >> ~/.bashrc
ENV PATH=/opt/conda/bin:$PATH

# Add the repository to the image
ADD . /workspace/csl/
