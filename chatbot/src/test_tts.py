#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
from voice.srv import tts as tts_srv
from voice.srv import vad as vad_srv

print("wait_for_service('/zordon/vad')")
rospy.wait_for_service('/zordon/vad')
print("wait_for_service('/zordon/tts')")
rospy.wait_for_service('/zordon/tts')