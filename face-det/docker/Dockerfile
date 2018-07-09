FROM nvidia/cuda:9.1-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim \
        python3 python3-dev python3-pip python3-setuptools python3-wheel build-essential ffmpeg cmake libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /code

WORKDIR /code

ADD requirements.txt /code/

# Install any needed packages specified in requirements.txt
RUN pip3 install -r /code/requirements.txt
