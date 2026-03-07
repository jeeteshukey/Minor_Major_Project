import streamlit as st

st.set_page_config(page_title="Cattle Analysis System", layout="wide")

st.title("🐄 Cattle Analysis System")

st.write(
"""
This system uses AI to analyze cattle using multiple modules:

• Breed Classification  
• Pose Estimation  
• Gait Analysis  
• Facial Recognition  
"""
)

st.markdown("### Select a Module")

col1, col2 = st.columns(2)

with col1:
    if st.button("🐄 Breed Classification"):
        st.switch_page("pages/1_Breed Classifier.py")

    if st.button("🦴 Pose Estimation"):
        st.switch_page("pages/2_Pose Estimation.py")

with col2:
    if st.button("🚶 Gait Analysis"):
        st.switch_page("pages/3_Gait Analysis.py")

    if st.button("🙂 Facial Recognition"):
        st.switch_page("pages/4_Facial Recognition.py")