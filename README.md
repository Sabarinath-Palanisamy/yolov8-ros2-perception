[Finaloutput.webm](https://github.com/user-attachments/assets/9e667a39-59d9-4a7a-832d-8cf4f9c3905c)

# Roadwork Intention Perception 🚧🤖

![ROS2](https://img.shields.io/badge/ROS2-Jazzy-3498DB?style=flat-square&logo=ros)
![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-E95420?style=flat-square&logo=ubuntu)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9D00?style=flat-square)

This repository contains the core perception stack developed for the **Autonomous Vehicles Master's exercise at TH-OWL**. 

Designed to bridge the safety gap in volatile construction zones, this module ingests raw camera feeds and processes them through a custom-trained YOLOv8 machine learning pipeline. It upgrades stateless object detection into a stateful tracking architecture (via ByteTrack) to calculate motion vectors, identify worker compliance, and predict dynamic intentions, eventually publishing the telemetry as structured payloads via ROS2.

## ✨ Key Features
* **Multi-Class Detection:** A unified 11-class schema identifying workers, PPE (hardhats/vests), traffic cones, signage, and heavy machinery.
* **Persistent Tracking:** Integration of the ByteTrack algorithm to assign and hold unique IDs for moving agents across consecutive frames.
* **Instance Segmentation (Upcoming):** Pixel-level polygon masking to perfectly isolate dynamic agents from static infrastructure.
* **ROS2 Architecture:** Built as a VDI-ADC compliant microservice (`vdi_brain`) utilizing a publisher/subscriber node network.

## 🛠️ System Requirements
* **OS:** Ubuntu 24.04 LTS
* **Framework:** ROS2 Jazzy
* **Language:** Python 3.12+
* **Hardware:** Tested on Intel Core Ultra 5 (CPU Inference). GPU acceleration supported via PyTorch CUDA.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/roadwork-perception-module.git](https://github.com/Sabarinath-Palanisamy/roadwork-perception-module.git)
   cd roadwork-perception-module
   ```

2. **Set up the isolated Python environment:**
   ```bash
   python3 -m venv yolo8_env
   source yolo8_env/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install ultralytics opencv-python
   ```

## 🎯 Usage (Inference & Tracking)

To test the compiled neural network weights and the ByteTrack ID assignment logic on a live camera feed:

```bash
# Ensure your environment is activated
source yolo8_env/bin/activate

# Run the tracker script
python live_tracker.py
```
*(Press `q` to safely exit the live camera window).*

## 📊 Dataset & Schema
This model was trained on a custom aggregated master dataset merged from 5 independent sources. *(Note: Due to size constraints, the raw image dataset is not hosted in this repository).*

**The 11-Class Schema:**
`hardhat`, `mask`, `no-hardhat`, `no-mask`, `no-vest`, `person`, `safety-cone`, `vehicle`, `vest`, `road-sign`, `road-marking`

## 🗺️ Project Roadmap
- [x] **Milestone 1:** Literature review, Master Data Merge, & Pipeline Verification.
- [ ] **Milestone 2 (June 08):** Finalization of trained YOLOv8-seg network and ROS2 node structure.
- [ ] **Milestone 3 (June 22):** Tracking integration and Motion Vector (Δpos/Δt) logic implementation.
- [ ] **Final (July 06):** Demonstration of Intention Recognition system & Final Report defense.

## 👨‍💻 Authors
- **Vignesh Marimuthu** Master of Information Technology | Autonomous Vehicles Technische Hochschule Ostwestfalen-Lippe (TH-OWL), Germany
- **Kashish Sharma** Master of Information Technology | Autonomous Vehicles Technische Hochschule Ostwestfalen-Lippe (TH-OWL), Germany
- **Anand** Master of Information Technology | Autonomous Vehicles Technische Hochschule Ostwestfalen-Lippe (TH-OWL), Germany
- **Sabarinath Palanisamy** Master of Information Technology | Autonomous Vehicles Technische Hochschule Ostwestfalen-Lippe (TH-OWL), Germany
