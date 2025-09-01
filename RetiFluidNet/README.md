# RetiFluidNet — Self-Adaptive Multi-Attention U-Net for Retinal OCT Fluid Segmentation

**Paper:** Rasti et al., *IEEE Transactions on Medical Imaging*, 2023 (doi:10.1109/TMI.2022.3228285)  
**Code (upstream):** https://github.com/aidialab/RetiFluidNet

RetiFluidNet is a convolutional architecture for **multi-class retinal fluid segmentation** (IRF, SRF, PED) in OCT B-scans. It combines a **Self-Adaptive Dual-Attention (SDA)** module, **Self-Adaptive attention-based Skip Connections (SASC)**, and a **multi-scale Deep Self-supervision Learning (DSL)** scheme, optimized with a joint loss (weighted Dice + edge-preserving connectivity terms). The model was validated on **RETOUCH**, **OPTIMA**, and **DUKE** public datasets. :contentReference[oaicite:0]{index=0}

## Setup

Create a Python environment (example with Conda):

```bash
conda create -n retifluidnet python=3.8 -y
conda activate retifluidnet

# TensorFlow 2.4 per the upstream repo
pip install "tensorflow==2.4.*"

# Common scientific stack (adjust as needed)
pip install numpy scipy scikit-image scikit-learn matplotlib tqdm opencv-python

## Data

RetiFluidNet was evaluated on public retinal OCT datasets; the RETOUCH challenge dataset is commonly used for IRF/SRF/PED segmentation. You must obtain data separately and comply with their licenses. 
retouch.grand-challenge.org
+1

A typical slice-based layout (customize to your dataset):

RetouchData/
├── Spectralis/
│   ├── retouch_data/          # B-scan images (e.g., PNG/JPG)
│   └── retouch_data_mask/     # label masks {0=BG,1=IRF,2=SRF,3=PED}
└── Cirrus/
    ├── retouch_data/
    └── retouch_data_mask/


Edit paths in train.py (and DataReader.py if your naming differs).

