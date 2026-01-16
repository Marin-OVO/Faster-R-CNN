# Remote Sensing Small Object Detection Experiment Code Description

This repository is a **course experiment codebase**, mainly used for reproducing and comparing methods related to **small object detection in remote sensing images**.  
The overall implementation is based on the **PyTorch** framework, and the experiments are conducted around the **Faster R-CNN + FPN** model.

It should be explicitly noted that **the dataset is NOT included in this repository and has been intentionally removed**.  
Before using this code, please **prepare the AITODv2 dataset by yourself**.

## 1. Runtime Environment

This project is developed and tested under **Python 3.11**.  
It is recommended to use **Conda** or a virtual environment for dependency management.

- Python: 3.11.7  
- Operating System: Windows  
- GPU: NVIDIA GPU  
- CUDA: 12.1  

## 3. Dependencies

The following libraries are **mandatory dependencies** for running and training the code.  
Missing any of them may cause runtime errors or training failures.

- torch == 2.1.0+cu121  
- torchvision == 0.16.0+cu121  
- numpy == 1.26.4  
- opencv-python == 4.12.0.88  
- pycocotools == 2.0.10  
- albumentations == 2.0.8  
- matplotlib == 3.7.5  
- tqdm == 4.67.1  
- pyyaml == 6.0.1  
- pandas == 2.0.3  
- scipy == 1.10.1  
- scikit-learn == 1.3.2  
- pillow == 9.5.0  

## 4. Usage Instructions

- Create and activate a Python virtual environment  
- Install the required **core dependencies** listed above  
- Download and configure the dataset paths manually  
- Check the dataset paths and training parameters in the configuration files  
- Run the training or testing scripts  
