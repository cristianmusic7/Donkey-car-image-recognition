
#Docker file for a slim Ubuntu-based Python3 image

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get update

RUN apt-get install -y python3-numpy python3-cffi python3-aiohttp \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev libopus-dev \
    libvpx-dev pkg-config pulseaudio

RUN pip3 install rtcbot

RUN apt-get install -y libsrtp2-dev python3-opencv

RUN apt-get install -y vim

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    x11vnc \
    xvfb \
    xterm \
    fluxbox && \
    apt-get clean

# Create and configure the VNC user
ENV VNCPASSWORD 1234
ARG VNCPASS
ENV VNCPASS ${VNCPASS:-secret}

RUN useradd remote --create-home --shell /bin/bash --user-group --groups adm,sudo && \
    echo "remote:$VNCPASS" | chpasswd
    
RUN pip install tensorflow

RUN pip install matplotlib

RUN pip3 install keras --upgrade

RUN apt-get install -y git

RUN git clone --recurse-submodules https://github.com/tensorflow/models.git 

RUN apt-get install curl unzip wget

# Make sure you grab the latest version
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip

# Unzip
RUN unzip protoc-3.5.1-linux-x86_64.zip -d protoc3

# Move protoc to /usr/local/bin/
RUN mv protoc3/bin/* /usr/local/bin/

# Move protoc3/include to /usr/local/include/
RUN mv protoc3/include/* /usr/local/include/

RUN mv /models/ app/

RUN wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz \
    && tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

RUN pip install opencv-contrib-python \
    && pip install zmq \
    && pip install imutils

COPY . /app

WORKDIR /app

RUN chmod +x main.sh

WORKDIR /app/research

RUN protoc object_detection/protos/*.proto --python_out=.

RUN pip install pillow

RUN pip uninstall tensorflow -y

RUN pip install tensorflow==1.5.0

WORKDIR /

RUN cp -R  ssdlite_mobilenet_v2_coco_2018_05_09 /app/research/object_detection/

EXPOSE 5555
EXPOSE 80
EXPOSE 5900


#ENTRYPOINT ["./main.sh"]
    
#CMD python desktop.py
