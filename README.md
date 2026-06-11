# AI-Based Low-Light Video Enhancement and Anomaly Detection System

## Overview

This project presents an AI-powered surveillance system designed to enhance low-light video footage and detect anomalous activities in real time. The system leverages deep learning techniques to improve visibility in poorly illuminated environments and applies anomaly detection algorithms for intelligent monitoring and security applications.

## Features

- Low-light video enhancement using Zero-DCE
- Real-time video processing
- Anomaly detection in surveillance footage
- Fire detection module
- Weapon detection support
- IP camera integration
- Improved visibility in challenging lighting conditions
- Automated monitoring and alert generation

## Technologies Used

- Python
- OpenCV
- PyTorch
- YOLO
- Zero-DCE
- NumPy
- Firebase
- Jupyter Notebook

## Project Structure

```text
AI-Based-Low-Light-Video-Enhancement-and-Anomaly-Detection/
│
├── Full/
│   ├── video.py
│   ├── video_full_fire.py
│   ├── ip_cam_full.py
│   ├── model.py
│   └── other detection modules
│
├── ZeroDCE/
│   ├── model.py
│   ├── image.py
│   ├── video.py
│   └── zerodce.pth
│
├── P69_Low_Light.ipynb
└── README.md
```

## System Workflow

1. Capture or load low-light video input.
2. Extract video frames.
3. Enhance frames using the Zero-DCE model.
4. Perform object and anomaly detection on enhanced frames.
5. Generate alerts for suspicious activities.
6. Display and store processed results.

## Installation

### Clone the Repository

```bash
git clone https://github.com/Abhinandpv10/AI-Based-Low-Light-Video-Enhancement-and-Anomaly-Detection.git
```

### Navigate to Project Directory

```bash
cd AI-Based-Low-Light-Video-Enhancement-and-Anomaly-Detection
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main surveillance pipeline:

```bash
python Full/video.py
```

Run IP camera monitoring:

```bash
python Full/ip_cam_full.py
```

## Applications

- Smart Surveillance Systems
- Night-Time Security Monitoring
- Traffic Monitoring
- Public Safety Systems
- Industrial Security
- Intelligent Video Analytics

## Results

The system significantly improves visibility in low-light environments while preserving important scene details. Enhanced video quality improves the effectiveness of anomaly detection and enables more accurate identification of suspicious activities, safety threats, and security incidents.

## Future Scope

- Edge AI deployment for real-time monitoring
- Multi-camera surveillance integration
- Transformer-based anomaly detection models
- Cloud-based monitoring dashboard
- Advanced threat classification and alert systems

## Author

**Abhinad Krishnan P V**

- GitHub: https://github.com/Abhinandpv10
- LinkedIn: linkedin.com/in/abhinand-krishnan-pv

## License

This project is developed for academic and research purposes only.

## Acknowledgements

- Zero-DCE for low-light image enhancement
- YOLO for object detection
- OpenCV for computer vision processing
- PyTorch for deep learning implementation