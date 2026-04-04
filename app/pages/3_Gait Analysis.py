import streamlit as st
import cv2
import tempfile
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your lameness detection module (you'll need to create this)
from src.cow_screening.predict import predict_lameness_from_video


st.set_page_config(
    page_title="Cattle Lameness Detection",
    page_icon="",
    layout="wide"
)

st.title("Cattle Lameness Detection System")
st.markdown("Upload a video of a walking cow to detect signs of lameness or tiredness")

# Main content area - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video of a walking cow"
    )
    
    if uploaded_file is not None:
        # Save uploaded video to temporary file
        file_extension = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # Display video player
        st.video(video_path)
        
        # Show video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        st.info(f"Video Info: {duration:.1f} seconds | {fps:.1f} fps | {frame_count} frames")

with col2:
    st.subheader("Analysis Results")
    
    if uploaded_file is not None:
        if st.button("Analyze Lameness", type="primary", use_container_width=True):
            with st.spinner("Analyzing gait patterns... This may take a moment..."):
                try:
                    # Call your lameness prediction function
                    # This should return: lameness_score, is_lame, confidence, gait_metrics
                    result = predict_lameness_from_video(
                        video_path,
                        confidence_threshold=confidence_threshold,
                        mode=gait_analysis_mode.lower()
                    )
                    
                    # Display results
                    st.markdown("### Results")
                    
                    # Show lameness status with color coding
                    if result['is_lame']:
                        st.error(f"Lameness Detected**")
                    else:
                        st.success(f"**Normal Gait**")
                    
                    # Show confidence and score
                    st.markdown(f"**Lameness Score:** {result['lameness_score']:.2f}/100")
                    st.markdown(f"**Confidence:** {result['confidence']:.1f}%")
                    
                    # Gait metrics in expandable section
                    with st.expander("Detailed Gait Metrics"):
                        metrics = result.get('gait_metrics', {})
                        
                        col_m1, col_m2 = st.columns(2)
                        
                        with col_m1:
                            st.metric("Stride Length", f"{metrics.get('stride_length', 'N/A')} pixels")
                            st.metric("Stance Duration", f"{metrics.get('stance_duration', 'N/A')} s")
                            st.metric("Head Bobbing", f"{metrics.get('head_bobbing', 'N/A')} pixels")
                        
                        with col_m2:
                            st.metric("Walking Speed", f"{metrics.get('speed', 'N/A')} pixels/s")
                            st.metric("Swing Duration", f"{metrics.get('swing_duration', 'N/A')} s")
                            st.metric("Back Posture", f"{metrics.get('back_posture', 'N/A')}°")
                    
                    # Show keypoint visualization (if available)
                    if 'keypoint_image' in result:
                        st.image(result['keypoint_image'], caption="Skeleton Keypoints Detected", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please ensure the video shows a cow clearly and try again")



