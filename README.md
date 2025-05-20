# Amazon Food Review Summarization with Transformers

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Transformer-based text summarization project for CSCI 3832 Spring 2025, analyzing Amazon Fine Food Reviews.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Preparation](#dataset-preparation)
- [Models](#models)
  - [RNN Model](#rnn-model)
  - [BART Model](#bart-model)
  - [T5 Model](#t5-model)
- [Results](#results)
- [Contributors](#contributors)

## Project Overview
This project implements three different models for generating summary insights from Amazon food reviews:
- **RNN** (Recurrent Neural Network)
- **BART** (Bidirectional and Auto-Regressive Transformers)
- **T5** (Text-to-Text Transfer Transformer)

## Dataset Preparation
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
2. Place `Reviews.csv` in data directory
3. Run preprocessing script:
```bash
python preprocesser.py
```
Generates `FilteredReviews.csv` used by all models.

## Models

### RNN Model
**Implementation**: `ForwardRNN.py`

### BART Model
**Implementation**: `bart_summarizer.ipynb`

#### Requirements:
```bash
conda create -n bart_env python=3.10
conda install pytorch=2.6.0 -c pytorch
pip install transformers==4.50 datasets evaluate kagglehub==0.3.8 rouge-score
```

#### Execution:
1. Place notebook in noptebooks directory
2. Ensure dataset path: `kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/Reviews.csv`
3. Run all notebook cells

### T5 Model
**Implementation**: `notebooks/t5_pretrained.ipynb`

#### Setup:
```bash
mkdir -p ../models ../data
cp FilteredReviews.csv ../data/
```

#### Requirements:
Same as BART model with additional space requirements for model caching

## Results

### BART Metrics
| Metric          | Pre-trained | Fine-tuned |
|-----------------|-------------|------------|
| **ROUGE-L**     | 0.0711      | 0.1335     |

### T5 Metrics
| Metric          | Pre-trained | Fine-tuned |
|-----------------|-------------|------------|
| **ROUGE-L**     | 0.1027      | 0.1614     |
| **BLEU**        | 0.0059      | 0.0289     |
| **METEOR**      | 0.1401      | 0.0999     |

All models include example review outputs in their final execution steps.

## Contributors
- **Ryen Johnston**: Data preprocessing pipeline
- **Miles Zheng**: BART implementation & optimization
- **Tian Zhang**: RNN architecture & training
- **Mohamed Abdelmagid**: T5 fine-tuning & evaluation + Formatting Workflow