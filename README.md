Here's an updated README.md file with dataset links included:

# U-Net Skin Lesion Segmentation with Explainable AI

This project implements a U-Net model for skin lesion segmentation using the ISIC 2018 dataset, with additional explainability features including Class Activation Maps (CAM) and Layer-wise Relevance Propagation (LRP).

## Project Overview

The system performs:
- Skin lesion segmentation using a U-Net architecture
- Model interpretability using CAM and LRP techniques
- Performance evaluation with F1-score and classification metrics

## Dataset

The project uses the ISIC 2018 Challenge dataset:

### Official Sources:
- **Main Challenge Page**: [ISIC 2018 Challenge Website](https://challenge.isic-archive.com/landing/2018/)
- **Task 1 (Lesion Segmentation)**: 
  - Training Input Images: [Download (JPEG, 5.4GB)](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip)
  - Training Ground Truth: [Download (PNG, 1.1GB)](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip)

### Alternative Sources:
- **Kaggle Dataset**: [ISIC 2018 on Kaggle](https://www.kaggle.com/datasets/shonenkov/isic2018)
- **Direct S3 Links** (if above don't work):
  ```
  https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip
  https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip
  ```

Dataset Specifications:
- Training images: 2,594 JPEG images
- Ground truth masks: 2,594 PNG masks
- Image size: Original varies, resized to 256×256 in this project

## Key Features

### Model Architecture
- Custom U-Net implementation with encoder-decoder structure
- Input size: 256×256×3 (RGB images)
- Output: Binary segmentation mask (256×256×1)

### Explainability Methods
1. **Class Activation Maps (CAM)**
2. **Layer-wise Relevance Propagation (LRP)**

## Installation

```bash
git clone https://github.com/yourusername/skin-lesion-segmentation.git
cd skin-lesion-segmentation
pip install -r requirements.txt
```

## Usage

1. **Download and prepare dataset**:
```bash
python prepare_data.py --data_dir ./data
```

2. **Train the model**:
```bash
python train.py --data_dir ./data --epochs 20
```

3. **Generate explanations**:
```bash
python explain.py --image_path sample.jpg --method cam
```

## Results

Sample outputs are saved in:
- `results/segmentations/`
- `results/explanations/` 
- `results/metrics/`
