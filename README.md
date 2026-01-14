# 🧠 ECT-HDGCN  
**Euler Characteristic Transform for Skeleton-Based Action Recognition**

---

## 1. Motivation and Background

Graph Convolutional Networks (GCNs) have become the dominant paradigm for skeleton-based action recognition due to their ability to model structured human body topology. Recent works have shown that **topological information** can further improve performance beyond purely graph-based representations.

In particular, recent studies such as **Block-GCN** explicitly incorporate concepts from **Topological Data Analysis (TDA)** to enhance topology awareness in skeleton graphs:

- **Block-GCN (CVPR 2024)**  
  *Redefine Topology Awareness for Skeleton-Based Action Recognition*  
  https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf

Another strong and widely adopted baseline is:

- **HD-GCN (ICCV)**  
  *Adaptive Hyper-Graph Convolution Network for Skeleton-Based Action Recognition*  
  https://arxiv.org/pdf/2208.10741

HD-GCN achieves strong performance by hierarchically decomposing skeleton graphs and applying attention-guided hierarchy aggregation. It already includes **chest-centered coordinate normalization**, which is treated as part of the baseline in this project.

---

## 2. Project Idea

Inspired by the success of TDA-based methods (e.g., persistence diagrams in Block-GCN), this project explores a different topological representation:

> **Euler Characteristic Transform (ECT)**

Compared to persistence diagrams, ECT:
- Is computationally efficient
- Captures global topological structure across multiple directions
- Provides a vectorized representation suitable for neural networks

To the best of my knowledge, **ECT has not been previously applied to skeleton-based action recognition**.

The core question of this project is:

> *Can ECT provide complementary topological information to HD-GCN, and what is the simplest and most effective way to integrate it?*

---

## 3. Method

### 3.1 Baseline

- Backbone: **HD-GCN**
- Streams: Joint stream and Bone stream
- Coordinate system: **Chest-centered (inherited from HD-GCN)**

No modification is made to:
- The original HD-GCN graph structure
- Hierarchical decomposition strategy
- Attention-guided hierarchy aggregation

---

### 3.2 Euler Characteristic Transform (ECT)

In this project:

- Skeleton joints are treated as a **3D point cloud**
- No graph connectivity is used during ECT computation
- ECT is computed independently from the GCN backbone

**ECT Computation Overview:**

1. Represent skeleton joints as a 3D point cloud
2. Center the point cloud
3. Project points onto a small set of predefined 3D directions
4. Compute Euler characteristics across multiple thresholds
5. Map ECT curves into a fixed-dimensional feature vector using an MLP

---

### 3.3 Explored Integration Strategies

I systematically explored multiple ECT-based designs, including:

- ECT-guided topological edge construction
- ECT-based gating mechanisms
- Multi-scale temporal ECT
- Learnable projection directions
- **Simple feature-level ECT fusion (ECT-fusion)**

**Key empirical finding:**

> Increasing architectural complexity does not necessarily improve performance.  
> The most effective strategy is **simple ECT feature fusion**.

---

### 3.4 Direction Design

- A small number of **fixed, approximately orthogonal directions** is sufficient
- Learnable directions tend to overfit
- Proper initialization is more important than learnability

This confirms that **ECT does not require heavy parameterization** to be effective.

---

## 4. Experimental Results

### Benchmark Difficulty Note

**NTU RGB+D 60 is a highly saturated and competitive benchmark.**  
The baseline models already achieve near-perfect training accuracy, making further improvements on the test set particularly difficult.  
In this context, even modest performance gains are considered non-trivial.

---

### NTU RGB+D 60 — Cross-Subject  
**Chest-Centered, CoM = Chest**

All results are obtained using the official HD-GCN configuration files:
- `joint_com_1.yaml`
- `bone_com_1.yaml`

| Method | HD-GCN Paper | This Project |
|------|--------------:|-------------:|
| Joint Stream | 90.4 | **90.57** |
| Bone Stream | 90.7 | **91.07** |
| Joint + Bone (Ensemble) | 92.4 | **92.61** |

Notably, ECT shows particularly strong improvements on the **Bone stream**, which focuses on geometric orientations and limb directions.

---

## 5. Reproducibility

This repository has been **fully verified**.  
Following the instructions below, all experiments (training, testing, and ensemble) can be reproduced.

---
## 🛠 Environment Setup (Verified)

The following environment configuration has been **fully verified to run successfully**.

### Core Environment

- Python: **3.9**
- CUDA: **12.8**
- PyTorch: **2.8.0 + cu128**
- torchvision: **0.23.0 + cu128**

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
````

---

### Additional Dependencies

```bash
pip install \
  scikit-learn \
  pyyaml \
  tensorboardX \
  tqdm \
  matplotlib \
  einops \
  h5py \
  packaging
```

---

### torchpack

* torchpack: **0.3.1**

```bash
pip install torchpack==0.3.1
```

> Note: `torchpack.runner` is not present in this version.
> Related logging components are treated as optional and do not affect training.

---

### torchlight (Local Package)

`torchlight` is included as a local editable package.

```bash
pip install -e torchlight
```

> Note: Due to the repository structure, `DictAction` is imported from
> `torchlight.torchlight.util` in the main training script.

---

## 📂 Data Preparation

### NTU RGB+D 60 

Request access:
[https://rose1.ntu.edu.sg/dataset/actionRecognition](https://rose1.ntu.edu.sg/dataset/actionRecognition)

Download:

* `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)

Extract to:

```text
./data/nturgbd_raw/
```

Example:

```bash
unzip nturgbd_skeletons_s001_to_s017.zip -d ./data/nturgbd_raw/
```

---


## ⚙️ Data Processing

### NTU RGB+D 60

```bash
cd ./data/ntu
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

⚠️ **Note**:
Running `seq_transformation.py` may be terminated with `Killed` due to high memory usage.

---

## 🏋️ Training

### NTU RGB+D 60 (Cross-Subject)

Joint stream:

```bash
python main.py \
  --config ./config/nturgbd-cross-subject/joint_com_1.yaml \
  --device 0
```

Bone stream:

```bash
python main.py \
  --config ./config/nturgbd-cross-subject/bone_com_1.yaml \
  --device 0
```

---

## 🔗 Ensemble Evaluation

```bash
python ensemble.py \
  --dataset ntu/xsub \
  --joint-pkl <joint_score.pkl> \
  --bone-pkl <bone_score.pkl>
```

---

## 📝 Summary

This project demonstrates that Euler Characteristic Transform (ECT) provides meaningful and complementary topological information for skeleton-based action recognition.

Through systematic experimentation, it shows that:

Complex architectural changes are not always beneficial

Simple ECT feature fusion is stable, effective, and parameter-efficient

Even on a highly saturated benchmark, ECT leads to consistent improvements

Overall, this project highlights ECT as a promising and lightweight topological tool for graph-based action recognition.


