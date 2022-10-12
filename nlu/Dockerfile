FROM dustynv/ros:melodic-ros-base-l4t-r32.7.1

WORKDIR /workspace

COPY . ./

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN apt install -y build-essential gfortran libatlas-base-dev python3.6-dev
RUN apt-get install -y pkg-config libpng-dev libfreetype6-dev libjpeg8-dev zlib1g-dev
RUN apt-get install -y python3-matplotlib python3-numpy python3-future python3-ujson python3-markupsafe
RUN apt-get install -y python3-scipy python3-h5py python3-pil
RUN python3 -m pip install cython pybind11 cppy
RUN python3 -m pip install matplotlib 
RUN python3 -m pip install tensorflow-aarch64==1.2

RUN python3 -m pip install -r roboga_nlu/requirements.txt
