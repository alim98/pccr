

# 1) OASIS — Inter-patient registration (Learn2Reg setup)

**Dataset:**

* OASIS MRI dataset
* 35 anatomical structures
* Learn2Reg 2021 evaluation protocol
* Train/Val/Test: 394 / 19 / 38 scans 

**Metrics:**

* Dice ↑
* HD95 ↓
* SDlogJ ↓

### Results

| Method     | Attention    | Dice ↑            | HD95 ↓            | SDlogJ ↓      |
| ---------- | ------------ | ----------------- | ----------------- | ------------- |
| VoxelMorph | –            | 0.847 ± 0.014     | 1.546 ± 0.306     | 0.133 ± 0.021 |
| DFI-NFF    | –            | 0.827 ± 0.013     | 1.722 ± 0.318     | 0.121 ± 0.015 |
| LapIRN     | –            | 0.861 ± 0.015     | 1.514 ± 0.337     | 0.072 ± 0.007 |
| ConvexAdam | –            | 0.846 ± 0.016     | 1.500 ± 0.304     | 0.067 ± 0.005 |
| TransMorph | Self-att     | 0.862 ± 0.014     | 1.431 ± 0.282     | 0.128 ± 0.021 |
| **H-ViT**  | Self + Cross | **0.876 ± 0.014** | **1.301 ± 0.264** | 0.539 ± 0.069 |

📌 نکته:

* SDlogJ برای H-ViT بدتره (0.539) → deformation rougher
* Dice و HD95 بهترین

---

# 2) IXI — Inter-patient & Patient-to-atlas

**Dataset:**

* IXI dataset
* 30 anatomical structures
* 115 pairs (inter-patient) / 150 pairs (atlas) 

**Metrics:**

* Dice ↑
* % non-positive Jacobian ↓ (folding)

---

## (A) Inter-patient

| Method | Dice ↑ | |JΦ| ≤ 0 (%) ↓ |
|-------|--------|--------------|
| VoxelMorph | 0.750 ± 0.106 | 1.013 ± 0.285 |
| MIDIR | 0.735 ± 0.093 | 0.295 ± 0.188 |
| CycleMorph | 0.750 ± 0.101 | 1.022 ± 0.293 |
| CoTr | 0.736 ± 0.112 | 0.702 ± 0.290 |
| nnFormer | 0.727 ± 0.101 | 1.284 ± 0.349 |
| PVT | 0.696 ± 0.116 | 1.868 ± 0.398 |
| ViT-V-Net | 0.772 ± 0.093 | 1.022 ± 0.289 |
| TransMorph-Bayes | 0.790 ± 0.081 | 1.136 ± 0.377 |
| TransMorph-Bspl | 0.793 ± 0.075 | < 0.001 |
| **H-ViT** | **0.810 ± 0.073** | 0.525 ± 0.209 |

---

## (B) Patient-to-atlas

| Method | Dice ↑ | |JΦ| ≤ 0 (%) ↓ |
|-------|--------|--------------|
| VoxelMorph | 0.734 ± 0.111 | 0.997 ± 0.197 |
| MIDIR | 0.722 ± 0.096 | 0.247 ± 0.107 |
| CycleMorph | 0.736 ± 0.105 | 0.992 ± 0.215 |
| CoTr | 0.717 ± 0.116 | 0.678 ± 0.205 |
| nnFormer | 0.718 ± 0.097 | 1.282 ± 0.256 |
| PVT | 0.690 ± 0.124 | 1.736 ± 0.248 |
| ViT-V-Net | 0.749 ± 0.102 | 1.033 ± 0.208 |
| TransMorph-Bayes | 0.772 ± 0.082 | 1.078 ± 0.236 |
| TransMorph-Bspl | 0.778 ± 0.080 | < 0.001 |
| **H-ViT** | **0.797 ± 0.075** | 0.565 ± 0.161 |

📌 نکته:

* H-ViT بهترین Dice
* TransMorph-Bspl بهترین topology (≈ بدون folding)

---

# 3) ADNI

**Dataset:**

* ADNI dataset
* 45 anatomical structures
* 150 pairs (inter + atlas) 

**Metric:**

* Dice ↑

---

## Results

| Method          | Inter-patient Dice ↑ | Atlas Dice ↑      |
| --------------- | -------------------- | ----------------- |
| Affine          | 0.531 ± 0.082        | 0.477 ± 0.052     |
| VoxelMorph      | 0.692 ± 0.214        | 0.646 ± 0.226     |
| MIDIR           | 0.666 ± 0.220        | 0.635 ± 0.227     |
| CycleMorph      | 0.687 ± 0.217        | 0.655 ± 0.225     |
| ViT-V-Net       | 0.727 ± 0.210        | 0.686 ± 0.219     |
| TransMorph-Bspl | 0.730 ± 0.208        | 0.702 ± 0.213     |
| **H-ViT**       | **0.760 ± 0.203**    | **0.730 ± 0.210** |

---

# 4) LPBA

**Dataset:**

* LPBA40 dataset
* 56 structures
* 120 / 117 pairs 

---

## Results

| Method          | Inter-patient Dice ↑ | Atlas Dice ↑      |
| --------------- | -------------------- | ----------------- |
| Affine          | 0.561 ± 0.018        | 0.543 ± 0.017     |
| CycleMorph      | 0.654 ± 0.017        | 0.645 ± 0.016     |
| nnFormer        | 0.626 ± 0.018        | 0.631 ± 0.016     |
| PVT             | 0.637 ± 0.016        | 0.642 ± 0.016     |
| ViT-V-Net       | 0.658 ± 0.017        | 0.650 ± 0.017     |
| TransMorph-Bspl | 0.670 ± 0.018        | 0.666 ± 0.016     |
| **H-ViT**       | **0.704 ± 0.016**    | **0.694 ± 0.015** |

---

# 5) Mindboggle

**Dataset:**

* Mindboggle dataset
* 41 structures
* 111 / 222 pairs 

---

## Results

| Method          | Inter-patient Dice ↑ | Atlas Dice ↑      |
| --------------- | -------------------- | ----------------- |
| Affine          | 0.537 ± 0.041        | 0.534 ± 0.034     |
| VoxelMorph      | 0.674 ± 0.197        | 0.666 ± 0.201     |
| CycleMorph      | 0.679 ± 0.194        | 0.671 ± 0.199     |
| CoTr            | 0.633 ± 0.214        | 0.630 ± 0.218     |
| ViT-V-Net       | 0.700 ± 0.186        | 0.695 ± 0.187     |
| TransMorph-Bspl | 0.699 ± 0.181        | 0.695 ± 0.183     |
| **H-ViT**       | **0.731 ± 0.170**    | **0.726 ± 0.173** |
