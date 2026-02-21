# Methodological Summary

This repository implements a subject-independent locomotion mode classification pipeline intended for embedded deployment. Only a concise overview is provided here; full justification belongs to the paper.

---
## Design Philosophy

The pipeline performs:

• Careful, safe and deterministic data preprocessing  
• Subject-independent LOSO evaluation on .tflite  
• MCU-safe deployment  

---

## Dataset Structure

Multiple subjects perform a fixed set of locomotion activities for two minutes each.

### - Augmentation

Training-only dataset augmentation emulate sensor and don/doff variability by:
• Per-sensor scaling  
• Offset shifts  
• Gaussian noise  

Augmentation is not applied to validation or test data.


### - Segmentation

Six Resistive Strain Sensor (one across every lower body joint) recordings are segmented into fixed-length temporal windows.
Segmentation parameters:

• Window Size (W)  
• Step Distance / Stride (S)

Each window represents a single training instance. 
Sliding-window segmentation is used to transform continuous sensor streams into supervised samples. Overlap is controlled via stride selection to balance:

• Temporal resolution  
• Sample count  
• Redundancy  

No label mixing is permitted within a window.

---

## Evaluation Protocol

Leave-One-Subject-Out (LOSO):

This estimate true cross-subject generalization rather than intra-subject performance. For N subjects:

• Train on N−1 subjects  
• Test on held-out subject  

---

## Model Family

Classifier Architecture:

• Two-layer 1D CNN + One-layer ANN  
• ReLU activations  
• Max + Global Avg Pooling (reduce params for MCU)  

---

## Quantization & Deployment

Selected models are converted using:

• Full INT8 quantization,  
• Representative dataset calibration,  
• TFLite Micro compatible ops,  

to preserve deployment fidelity between training and inference environments.


---

For detailed rationale, refer paper.
