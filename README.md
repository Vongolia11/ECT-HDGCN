# 🧠 ECT-HDGCN  
## Exploring Euler Characteristic Transform (ECT) on Top of HD-GCN

---

## 📌 Abstract

Graph Convolutional Networks (GCNs) are widely used for skeleton-based action recognition and have achieved strong performance.  
Among them, **HD-GCN** proposes a hierarchically decomposed graph structure that effectively models both structurally adjacent and distant joint relations, together with an attention-guided hierarchy aggregation module.

In this course project, **HD-GCN is taken as a fixed baseline**.  
The goal of this project is **not to redesign the graph structure**, but to explore whether **topological features derived from Euler Characteristic Transform (ECT)** can further improve skeleton-based action recognition, and **how ECT should be integrated in practice**.

Through extensive experiments with different ECT designs, I find that a **simple ECT feature fusion strategy (ECT-fusion)** achieves the best and most stable performance, outperforming more complex alternatives.

---

## 🔍 Background and Baseline

- **Baseline Model**: HD-GCN (Hierarchically Decomposed Graph Convolutional Network)
- **Key Components of HD-GCN**:
  - Hierarchically Decomposed Graph (HD-Graph)
  - Attention-guided Hierarchy Aggregation (A-HA)
  - Joint and Bone streams
  - Chest-centered coordinate normalization (original design of HD-GCN)



---

## 🧪 Scope of My Contribution

This project focuses exclusively on **Euler Characteristic Transform (ECT)** and its role in skeleton-based action recognition.

I systematically explored multiple ECT-related designs, including:

- 🔗 ECT-guided topological edge construction
- 🎛️ ECT-based gating mechanisms
- ⏱️ Multi-scale temporal ECT modeling
- 🎓 Learnable ECT projection directions
- ➕ Feature-level ECT fusion (**ECT-fusion**)

After extensive experimentation, I conclude that:

> ⭐ **A simple ECT-fusion strategy is the most effective and robust design.**

---

## 🧩 Euler Characteristic Transform (ECT)

ECT is a topological descriptor that captures **global structural properties** of a shape.

In this project:

- **Skeleton joints are treated as a 3D point cloud**
- No graph connectivity or bone topology is used during ECT computation
- ECT is computed independently from the skeleton graph

This design makes ECT **complementary** to graph-based modeling in HD-GCN.

---

## ⚙️ ECT Computation (Code-Level Description)

For each input sequence:

1. Skeleton joints are represented as a 3D point cloud for each frame  
2. The point cloud is centered by subtracting the mean joint position  
3. Points are projected onto a small set of predefined 3D directions  
4. For each direction, Euler characteristics are computed across multiple thresholds  
5. The resulting ECT curves are mapped by an MLP into a fixed-dimensional feature vector  

ECT features are computed **in parallel** with the HD-GCN backbone.

---

## 🧭 Projection Directions: Initialization vs. Learning

An important empirical finding of this project concerns ECT projection directions.

### Observations

- A **small number of fixed, approximately orthogonal directions** is sufficient
- Making directions learnable does **not** improve performance
- Learnable directions often lead to instability or overfitting

### Final Choice

- Directions are initialized using uniformly distributed vectors on the sphere
- Directions are kept **fixed during training**

This simple design is both **stable and effective**, especially for skeleton graphs.

---

## 🏁 Final Design: ECT-Fusion

After comparing all variants, the final design adopts **ECT-fusion**:

- HD-GCN extracts local spatio-temporal features
- ECT extracts global topological features
- The two representations are fused by **simple feature addition**

This avoids unnecessary architectural complexity while preserving complementary information.

---

## 📊 Experimental Results  
**NTU RGB+D 60 — Cross-Subject (xsub)**

| Method | HD-GCN Paper | This Project |
|------|--------------:|-------------:|
| Joint Stream | 90.4 | **90.57** |
| Bone Stream | 90.7 | **91.07** |
| Joint + Bone (Ensemble) | 92.4 | **92.61** |

All results are obtained under the same evaluation protocol.

---

## 💡 Key Takeaways

- ECT provides complementary **global topological information**
- Complex ECT-based designs are not necessarily better
- **ECT-fusion** is the simplest and most robust strategy
- A small set of fixed directions is sufficient for skeleton topology

---

## 📝 Summary

This course project systematically explores **Euler Characteristic Transform (ECT)** on top of HD-GCN.

By evaluating multiple ECT integration strategies, I demonstrate that a **simple ECT-fusion design** is both effective and efficient, improving upon a strong HD-GCN baseline without modifying its core architecture.
