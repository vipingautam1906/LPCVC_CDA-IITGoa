
# Team CDA-IITGoa — LPCVC 2026 Submission

This repository contains the official submission by **Team CDA-IITGoa** for the **IEEE Low Power Computer Vision Challenge (LPCVC) 2026**.

## 🧠 Overview

We present an efficient video understanding pipeline designed to operate under strict **latency and accuracy constraints** defined by LPCVC.

- 🚀 **Target Device:** Dragonwing IQ-9075 EVK  
- ⚡ **Goal:** Maximize accuracy while meeting real-time inference requirements  
- 🧩 **Approach:** Training-focused improvements without modifying the base architecture  



## 🏋️ Model Training

- 🎬 **Backbone:** R(2+1)D-18 video model  
- 🔢 **Output Layer:** Adapted for **92 classes**  
- 🔓 **Training Strategy:** Full model fine-tuning (all layers trainable)  
- ⚙️ **Framework:** PyTorch  



## 🔬 Training Strategy

We observed that the standard sample solution lacks sufficient generalization under strict latency constraints.  

Instead of modifying the architecture, we focused on:

- 📈 Improving generalization through **data-centric strategies**  
- 🎯 Careful **fine-tuning of the entire network**  
- ⚖️ Maintaining a strong **accuracy–latency balance**



## 🎥 Augmentation Pipeline

To enhance robustness and performance, we employ a probabilistic augmentation stack:

- 🔀 **VideoMix** — *60% probability*  
  - Mixes video clips to improve temporal generalization  

- 🧱 **StackMix** — *10% probability*  
  - Combines spatial-temporal segments across samples  

- 🎞️ **VideoColorJitter** — *50% probability*  
  - Introduces temporal variation for better robustness  


## ⚙️ Main Script

- **Core Script:** `train-videoMix.py`  
- Used for both **training and evaluation**  
- Use our best weights (url given below) in "--test-only" mode to eval
- Behavior controlled via command-line arguments (`argparse`)  


### Train & Eval
#### Train
```bash
python references/video_classification/train-videoMix.py \
    --data-path /path/to/dataset \
    --output-dir /path/to/save/checkpoints \
    --epochs 50 \
    --batch-size 64 \
    --amp
```

#### Eval 
To reproduce our results, download the reported weights from the provided URL and place the file at:
```bash
mkdir weights
weights/model_4.pth
```
Run below to finaly evaluate 

```bash
python references/video_classification/train-videoMix.py \
    --data-path /path/to/dataset \
    --output-dir /path/to/save/checkpoints \
    --epochs 50 \
    --batch-size 64 \
    --resume weights/model_4.pth \
    --amp \
    --test-only
```

### 📤 Export (Qualcomm AI Hub)

After training, export the best checkpoint for deployment on **Dragonwing IQ-9075 EVK**.

The script below handles:
- Model loading (PyTorch checkpoint)
- Conversion to deployment format
- Compilation for target device
- Upload and optimization via Qualcomm AI Hub

The final **.bin (QNN context) model** is generated inside `export_assets/`.

> ⚠️ **Important:**  
> Ensure the correct path to the trained PyTorch checkpoint is set before running export:
>
> ```python
> ckpt_path = "weights/model_4.pth"
> ```



### 🚀 Run Export

```bash
python export.py 
```

## 📦 Pretrained Models & Downloads

We provide both the **trained PyTorch checkpoint** and the **exported QNN (.bin) model** for reproducibility and direct deployment.

These correspond to our **final submission used for LPCVC 2026 evaluation**.

---

### 🔗 Download Links

- 🧠 **PyTorch Trained Weights**  
  Pretrained model after full fine-tuning with our augmentation pipeline:  
  👉 https://drive.google.com/file/d/16Tdjfm28kcaJ5wFoo3yr0Pa1NMHjRmUn/view?usp=sharing

- ⚡ **QNN Context Binary (.bin)**  
  Fully compiled model for **Dragonwing IQ-9075 EVK** (ready for inference):  
  👉 https://drive.google.com/file/d/1Wu2G5GaRm-LKVytPFZnbCqdCplHj3pSZ/view?usp=sharing


## 🙏 Acknowledgement

Our implementation builds upon the official LPCVC sample solution provided by the organizers.

For environment setup, dependencies, and base execution details, please refer to the official repository:
```
🔗 https://github.com/lpcvai/26LPCVC_Track2_Sample_Solution
```
We follow the same setup procedure and extend it with our training strategy, augmentations, and deployment pipeline.We would like to sincerely thank the **LPCVC 2026 organizing team** Providing a well-structured and sample solution.

