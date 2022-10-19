FROM dustynv/ros:melodic-ros-base-l4t-r32.7.1

WORKDIR /workspace

RUN apt-get update
RUN apt-get update
RUN apt -y install nano
RUN apt -y install  libpng-dev libfreetype6-dev python3.6-dev
RUN apt -y install python3.6 python3.6-dev
RUN apt-get -y install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
RUN apt-get -y install python3-pip
RUN python3 -m pip install -U pip testresources setuptools==49.6.0
RUN python3 -m pip install pyaml 
RUN python3 -m pip install sklearn_crfsuite==0.3.6 
RUN wget https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-1.15.5+nv21.12-cp36-cp36m-linux_aarch64.whl
RUN apt-get -y install python3-protobuf
RUN python3 -m pip install protobuf==3.6.1 Cython
RUN apt-get -y install python3-h5py python3-numpy python3-termcolor
RUN python3 -m pip install pip install tensorflow-1.15.5+nv21.12-cp36-cp36m-linux_aarch64.whl
RUN apt-get -y install python3-scipy
RUN python3 -m pip install numpy==1.18 scikit-learn==0.20.4 
RUN python3 -m pip install rasa-nlu==0.14.3 
RUN python3 -m pip install rospkg
RUN python3 -m pip install -U pip setuptools wheel
RUN python3 -m pip install -U scipy
COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

RUN mkdir -p src/interacao_nlu
COPY . ./src/interacao_nlu
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'

RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc 
RUN echo "source /workspace/devel/setup.bash" >> /root/.bashrc 
COPY ros_entrypoint.sh /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["roslaunch", "interacao_nlu", "rasa_nlu.launch"]
