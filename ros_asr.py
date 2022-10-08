import rospy
from std_srvs.srv import Trigger, TriggerResponse

if __name__ == "__main__":
    def handler(req):
        print(req)
        text = client.activate()
        return TriggerResponse(
            success=True,
            message=text
        )
    rospy.init_node('speech_to_text', anonymous=True)
    service = rospy.Service('roboga/stt', Trigger, handler)    
    
    rospy.spin()