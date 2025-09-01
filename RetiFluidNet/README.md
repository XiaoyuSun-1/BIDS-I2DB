# RetiFluidNet — Self-Adaptive Multi-Attention U-Net for Retinal OCT Fluid Segmentation

**Paper:** Rasti, Reza et al. “RetiFluidNet: A Self-Adaptive and Multi-Attention Deep Convolutional Network for Retinal OCT Fluid Segmentation.” IEEE transactions on medical imaging vol. 42,5 (2023): 1413-1423. doi:10.1109/TMI.2022.3228285
 
**RetiFluidNet original github:** https://github.com/aidialab/RetiFluidNet

## Environment setup 


```bash
cd RetiFluidNet
conda env create -f environment.yml
conda activate retifluidnet   # if your environment.yml uses a different name, activate that

## Data Layout

Add a RetouchData folder under this RetifluidNet folder with the following structure.

RetouchData/
├── Spectralis/
│   ├── raw_bscans/            # The raw bscans from the RETOUCH dataser
│   └── raw_masks/  
│   └── retouch_data/          # OCT B-scans (PNG/JPG)
│   └── retouch_data_mask/     # masks {0=BG, 1=IRF, 2=SRF, 3=PED}
└── Cirrus/
    ├── raw_bscans/
    └── raw_masks/  
    └── retouch_data/          # OCT B-scans (PNG/JPG)
    └── retouch_data_mask/     # masks {0=BG, 1=IRF, 2=SRF, 3=PED}
