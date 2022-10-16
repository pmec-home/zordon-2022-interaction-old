FROM dustynv/ros:melodic-ros-base-l4t-r32.7.1

WORKDIR /workspace

RUN apt-get update
RUN apt-get update
RUN apt -y install python3.7
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN apt -y install  libpng-dev libfreetype6-dev python3-dev
RUN apt -y install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
RUN apt -y install python3-pip
RUN python3 -m pip install -U pip testresources setuptools==49.6.0
RUN apt -y install python3-tk
RUN python3 -m pip install tk
COPY . ./

CMD ["./mainframe.py"]
ENTRYPOINT ["python3"]




