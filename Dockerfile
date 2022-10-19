FROM jetson-voice-w2v-whisper-ros

RUN mkdir -p /interacao_ws/src
COPY . /interacao_ws/src/
WORKDIR /interacao_ws
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make' && \ 
    echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc && \ 
    echo "source /interacao_ws/devel/setup.bash" >> /root/.bashrc 

# chatbot

RUN python3 -m pip install spacy==3.4.1 spacy download en_core_web_sm
# ===========================================================================================
RUN pip install netifaces 

COPY ros_entrypoint.sh /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["roslaunch", "chatbot", "chatbot.launch"]
