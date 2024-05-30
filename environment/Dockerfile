# hash:sha256:8e0c0a467012842bd122024cbf04a339a4526aa9e634866e962fa3f0203e95cf
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN mamba install -y \
        caiman==1.11.0 \
    && mamba clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.37.3 \
    dateparser==1.2.0 \
    jnormcorre==1.0.0 \
    matplotlib==3.9.0 \
    opencv-python-headless==4.9.0.80 \
    scanimage-tiff-reader==1.4.1.4 \
    scikit-image==0.23.2 \
    scipy \
    tenacity==8.3.0 \
    tifffile \
    -e git+https://github.com/calebshibu/suite2p.git@c88e1ba3aa2013fe379e64c7e45ae73a417eb4a7#egg=suite2p

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
