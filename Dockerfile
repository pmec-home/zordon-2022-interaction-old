FROM dustynv/jetson-voice:r32.7.1

# ===========================================================================================
# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# ===========================================================================================
RUN apt update
RUN apt install -y curl 
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
RUN apt install -y ros-melodic-ros-base
# ===========================================================================================

RUN python3 -m pip install --upgrade pip \
     && python3 -m pip install rospkg

# ===========================================================================================
# setup entrypoint
COPY ./ros_entrypoint.sh /

ENV ROS_DISTRO=melodic

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc 