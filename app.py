from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import numpy as np
import HandModule2
import av

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Video(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img=HandModule2.find_hands(img)
        #image gray
        return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="key", video_processor_factory=Video, rtc_configuration=RTC_CONFIGURATION)