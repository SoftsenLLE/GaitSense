# A Wearable Hyperelastic Strain Sensing Suit for Locomotion Classification
## Locomotion Mode Classification on ESP32-S3

This repository contains the complete training and embedded inference workflow for a locomotion mode / walking terrain classifier using microcontrollers and lower limb resistive strain sensors, built by researchers under HCRC Group at CSIR - CMERI, India 🇮🇳. 

The repository covers dataset construction, Leave-One-Subject-Out (LOSO) evaluation, CNN training, INT8 quantization, and real-time deployment using TensorFlow Lite Micro on ESP32-S3. It is structured to separate model development from embedded execution.

---

## Repository Folder Structure
```
├── embedded/
├── training_pipeline/
├ methodology
├ README
```


---

## Workflow Overview

1. Dataset construction and segmentation  
2. LOSO evaluation for subject-independent validation  
3. CNN training and hyperparameter study  
4. Full INT8 TFLite conversion  
5. Embedded inference on ESP32-S3  

All experimental design details and justification are provided in the accompanying paper.

---

## Training Pipeline

The notebooks implement:

• Controlled window/stride segmentation  
• Segment-wise dataset integrity checks  
• LOSO protocol  
• Data augmentation  
• CNN training  
• TFLite INT8 conversion  
• Metric aggregation and confusion matrices  

Execute notebooks sequentially.

---

## Embedded Inference

Two firmware variants are provided:

**locomotion_mode_sim**

Replays recorded data for verification and latency measurement.

**locomotion_mode_actualTest**

Runs live inference using ADC inputs (`analogRead()`), applying the quantization scale and zero values from training.

Both variants use Chirale_TensorFlowLite library (https://github.com/spaziochirale/Chirale_TensorFlowLite).

---

## Target Platform

ESP32-S3 (should work in other capable MCUs too)
TensorFlow Lite Micro  
INT8 quantized CNN  

Operator set is restricted to TFLM built-ins compatible with MCU deployment.

---

## Intended Scope

This repository is intended for research reproducibility and embedded ML demonstration. It is not packaged as a general-purpose framework.

For methodological rationale and experimental interpretation, refer to the paper: https://ieeexplore.ieee.org/abstract/document/xxxxxxxx
