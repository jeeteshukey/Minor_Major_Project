# Cattle Analysis System

## Overview

This project uses deep learning to analyze cattle images and videos.
Currently implemented module: **Breed Classification**.

The system is built with **PyTorch** for model training and **Streamlit** for the web interface.

## Features

* Breed classification of cattle from images
* Streamlit-based user interface
* Trained CNN model (ResNet18)

## Project Structure

app/ – Streamlit UI
src/ – Model training and prediction code
models/ – Trained model file (.pth)

## How to Run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Run the Streamlit app

```
streamlit run app/Home.py
```

3. Upload a cattle image to predict the breed.

## Detected Breeds

* Holstein Friesian
* Jaffarabadi
* Jersey
* Murrah
* Sahiwal
