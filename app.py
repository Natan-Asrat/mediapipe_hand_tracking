import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from hand_detector.module import HandTracking

class CustomVideoProcessor(VideoTransformerBase):
    def __init__(self, detector):
        self.detector = detector

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        drawn_image = self.detector.findHands(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main application logic
st.title("Mediapipe Hand Tracking Live Stream")
st.text("Live Hand Tracking  by Natan Asrat.")

# Load the YOLO model
@st.cache_resource(show_spinner=True)
def load_model():
    st.text("Loading MediaPipe Detector...")
    detector = HandTracking()
    return detector

detector = load_model()



# Use WebRTC for live video stream
webrtc_streamer(key="example",  video_processor_factory=lambda: CustomVideoProcessor(detector))
