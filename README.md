# CNT

## Overview
This repository contains the implementation of the CNT framework for the paper "Harnessing Contrastive Learning and Neural Transformation for Time Series Anomaly Detection".


## Getting Started

### Prerequisites
- Python >= 3.9

You can install all required dependencies using the following command:
```bash
pip install -r requirements.txt
```
### Datasets
The datasets used in our experiments can be downloaded from the following link:

Please place the datasets in the data/ directory.

### Running the code
To train and evaluate the CNT model on a specific dataset:
```
python train.py --datasets=msl,smap,swat,wadi --train
```









