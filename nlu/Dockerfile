FROM dustynv/ros:melodic-ros-base-l4t-r32.7.1

# nvcr.io/nvidia/l4t-tensorflow

RUN apt update

RUN apt install python3.8
RUN apt install python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN apt install -y build-essential gfortran libatlas-base-dev python3.8-dev
RUN apt install python3-scipy python3-h5py python3-pil
RUN apt install libjpeg8-dev zlib1g-dev
RUN python -m pip install cython pybind11 cppy 
RUN python -m pip install matplotlib
RUN python -m pip install -r requirements.txt 

WORKDIR /workspace

COPY . ./