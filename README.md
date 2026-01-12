
# 🧠 ECT-HDGCN  
## Exploring Euler Characteristic Transform (ECT) on Top of HD-GCN

---

## 📌 Abstract

Graph Convolutional Networks (GCNs) are widely used for skeleton-based action recognition and have achieved strong performance.  
Among them, HD-GCN introduces a hierarchically decomposed graph structure together with attention-guided hierarchy aggregation, enabling effective modeling of both structurally adjacent and distant joint relations.

In this course project, HD-GCN is taken as a fixed baseline.  
The goal is not to redesign the graph structure, but to explore whether topological features derived from Euler Characteristic Transform (ECT) can further improve skeleton-based action recognition, and how ECT should be integrated in a simple and effective manner.

Through extensive experiments, I find that a simple ECT feature fusion strategy (ECT-fusion) consistently outperforms more complex ECT-based designs.

---

## 🔍 Background and Baseline

- Baseline Model: HD-GCN (Hierarchically Decomposed Graph Convolutional Network)
- Key components:
  - Hierarchically Decomposed Graph (HD-Graph)
  - Attention-guided Hierarchy Aggregation (A-HA)
  - Joint and Bone streams
  - Chest-centered coordinate normalization

---

## 🧪 Scope of My Contribution

This project focuses exclusively on Euler Characteristic Transform (ECT).

I systematically explored multiple ECT-based designs, including:

- ECT-guided topological edge construction  
- ECT-based gating mechanisms  
- Multi-scale temporal ECT modeling  
- Learnable ECT projection directions  
- Simple feature-level ECT fusion (ECT-fusion)

After extensive experimentation, I conclude that:

> **A simple ECT-fusion strategy is the most effective and robust design.**

---

## 🧩 Euler Characteristic Transform (ECT)

ECT is a topological descriptor that captures global structural properties of a shape.

In this project:

- Skeleton joints are treated as a **3D point cloud**
- No graph connectivity or bone topology is used during ECT computation
- ECT is computed independently from the skeleton graph

This makes ECT complementary to graph-based modeling in HD-GCN.

---

## ⚙️ ECT Computation (Implementation Overview)

For each input skeleton sequence:

1. Skeleton joints are represented as a 3D point cloud for each frame  
2. The point cloud is centered by subtracting the mean joint position  
3. Points are projected onto a small set of predefined 3D directions  
4. For each direction, Euler characteristics are computed across multiple thresholds  
5. The resulting ECT curves are mapped by an MLP into a fixed-dimensional feature vector  

ECT features are computed in parallel with the HD-GCN backbone and fused at the feature level.

---

## 🧭 Projection Directions

An important empirical finding concerns ECT projection directions:

- A small number of fixed, approximately orthogonal directions is sufficient  
- Learnable directions do not improve performance  
- Fixed directions are more stable and efficient  

Directions are initialized using uniformly distributed vectors on the sphere and kept fixed during training.

---

## 🏁 Final Design: ECT-Fusion

The final design adopts **ECT-fusion**:

- HD-GCN extracts local spatio-temporal features  
- ECT extracts global topological features  
- The two representations are fused by simple feature addition  

This design avoids unnecessary architectural complexity while preserving complementary information.

---

## 📊 Experimental Results  
**NTU RGB+D 60 — Cross-Subject (Chest-Centered, CoM = Chest)**

All results are obtained using the official HD-GCN configuration files  
`joint_com_1.yaml` and `bone_com_1.yaml`, which adopt **chest-centered coordinate normalization**.

| Method | HD-GCN Paper | This Project |
|------|--------------:|-------------:|
| Joint Stream (`joint_com_1.yaml`) | 90.4 | **90.57** |
| Bone Stream (`bone_com_1.yaml`) | 90.7 | **91.07** |
| Joint + Bone (Ensemble) | 92.4 | **92.61** |

---

## 🔁 Reproducibility

All experiments are conducted by extending the official HD-GCN codebase and strictly following its original protocol.

---

## 📦 Dependencies

- Python ≥ 3.6  
- PyTorch ≥ 1.10.0  
- PyYAML == 5.4.1  
- torchpack == 0.2.2  
- matplotlib, einops, sklearn, tqdm, tensorboardX, h5py  

Install dependencies:

```bash
pip install -e torchlight
````

---

## 📂 Data Preparation

### Download Datasets

The following datasets are used:

* NTU RGB+D 60 Skeleton
* NTU RGB+D 120 Skeleton
* Northwestern-UCLA

**NTU RGB+D 60 **

Request access from:
[https://rose1.ntu.edu.sg/dataset/actionRecognition](https://rose1.ntu.edu.sg/dataset/actionRecognition)

Download skeleton-only data:

* nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)

Extract to:

```text
./data/nturgbd_raw/
```

## 🗂 Directory Structure

```text
data/
├── nturgbd_raw/
│   ├── nturgb+d_skeletons/
│   └── nturgb+d_skeletons120/
├── ntu/
├── ntu120/
└── NW-UCLA/
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
⚠️ **Note on Data Processing**

Running `seq_transformation.py` for NTU RGB+D 60 may result in the process being terminated (`Killed`) due to memory constraints, as the script allocates large intermediate arrays during data alignment.

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

To perform joint–bone ensemble evaluation:

```bash
python ensemble.py \
  --dataset ntu/xsub \
  --joint-pkl <joint_score.pkl> \
  --bone-pkl <bone_score.pkl>
```

The fusion weight is automatically searched and the best accuracy is reported.

---

## 📝 Summary

This course project systematically explores Euler Characteristic Transform (ECT) on top of HD-GCN.

By evaluating multiple ECT integration strategies, this work demonstrates that a simple ECT-fusion design is both effective and robust, improving performance over a strong HD-GCN baseline without modifying its core architecture.

```
