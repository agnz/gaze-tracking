FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
  python-lxml \
  python-pil \
  python-tk \
  git \
  && rm -rf /var/lib/apt/lists/*

# install protoc v3 (apt-get installs v2, but v3 is required)
RUN curl -LO https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip \
  && unzip protoc-3.5.1-linux-x86_64.zip -d protoc3 \
  && mv protoc3/bin/* /usr/local/bin/ \
  && mv protoc3/include/* /usr/local/include/ \
  && rm -r protoc3

WORKDIR /  
# clone the models repo
RUN git clone https://github.com/tensorflow/models.git

WORKDIR models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc
RUN python3 setup.py install

CMD ["/bin/bash"]
#CMD ["tail", "-f", "/dev/null"]
