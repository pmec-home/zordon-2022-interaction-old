FROM jetson-voice-w2v-whisper-ros

RUN mkdir -p /interacao_ws/src
COPY . ./interacao_ws/src/
WORKDIR /interacao_ws
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'

RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc 
RUN echo "source /interacao_ws/devel/setup.bash" >> /root/.bashrc 

# chatbot

RUN python3 -m pip install spacy==3.4.1
RUN python3 -m spacy download en_core_web_sm
# ===========================================================================================
RUN pip install netifaces 