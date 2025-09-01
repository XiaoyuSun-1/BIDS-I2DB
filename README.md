# One Size Fits All? Comparing Foundation and Task-specific Models for Retinal Fluid Segmentation

**RetiFluidNet (task-specific)** vs **VisionFM (foundation)** for segmenting intraretinal fluid (IRF), subretinal fluid (SRF), and pigment epithelial detachment (PED) on OCT B-scans. Using the RETOUCH benchmark, we build two reproducible pipelines: TensorFlow for RetiFluidNet and PyTorch for VisionFM, run 3-fold cross-validation across Spectralis and Cirrus devices, and evaluate pixel-level accuracy (Acc, BAcc, Dice, IoU) and patient and slice level agreement via fluid-proportion quantification and ICC. Overall, RFN delivers consistently stronger segmentation accuracy and more stable fluid quantification than VFM with qualitative examples illustrating cleaner fluid boundaries.

- **Pipelines:** Refer to the subfolders: RetiFluidNet for the task-specific model, and VisionFM for the foundation model. 
- **Dataset:** RETOUCH—Retinal OCT Fluid Challenge(https://retouch.grand-challenge.org/)  
- **Metrics:** Dice (DSC), Balanced Accuracy (BACC), ICC for fluid proportion (patient-sum, all-slices, fluid-only)  
- **Devices:** Spectralis & Cirrus; seed = 3407; 3-fold CV


## Results at a Glance

<p align="center">
  <img src="images/summary_tables.png" alt="Table 1: Dice and BACC for RFN vs VFM" width="70%">
  <br><em><strong>Table 1.</strong> Dice Similarity Coefficient (DSC) and Balanced Accuracy (BACC) for RFN and VFM.</em>
</p>

<p align="center">
  <img src="images/icc_fluid_proportion.png" alt="Figure 1: ICCs for fluid proportion (RFN left, VFM right)" width="85%">
  <br><em><strong>Figure 1.</strong> Intraclass correlation coefficients (ICC) for fluid proportion across devices and measures.</em>
</p>

<p align="center">
  <img src="images/visulizations.png" alt="Figure 2: OCT examples—GT vs RFN vs VFM (Spectralis & Cirrus)" width="85%">
  <br><em><strong>Figure 2.</strong> Example Spectralis (top) and Cirrus (bottom) slices with GT, RFN, and VFM segmentations.</em>
</p>

---

## Citation


## Acknowledgments
BIDS@I2DB Summer Research Internship, Institute for Informatics, Data Science & Biostatistics, Washington University School of Medicine.
All members from the CausAI lab.


