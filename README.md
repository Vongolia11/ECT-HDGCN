# 🧠 ECT-HDGCN  
## Exploring Euler Characteristic Transform (ECT) for Skeleton-Based Action Recognition 🦴

---

### 1. 📌 Background

Graph-based models such as **HD-GCN** achieve strong performance in skeleton-based action recognition by modeling spatial and temporal relations between joints and bones.

In the original HD-GCN framework, **chest-centered coordinate normalization** is already adopted.  
Therefore, **coordinate normalization itself is not a contribution of this project**.

This project focuses on a different research question:

> ❓ **How can Euler Characteristic Transform (ECT) be effectively integrated into skeleton-based action recognition, and what is the simplest yet most effective design?**

---

### 2. 🧪 Scope of My Contribution

My main contribution is a **systematic exploration of ECT-based representations and fusion strategies** on top of HD-GCN.

I experimented with multiple ways of incorporating ECT, including:

- 🔗 ECT-guided topological edge connections  
- 🎛️ ECT-based gating mechanisms  
- ⏱️ Multi-scale temporal ECT modeling  
- 🎓 Learnable ECT parameters  
- ➕ Simple feature-level ECT fusion (**ECT-fusion**)

Through extensive experiments, I found that:

> ⭐ **A simple ECT-fusion strategy consistently outperforms more complex designs.**

---

### 3. 🧩 Euler Characteristic Transform (ECT)

ECT is a topological descriptor that captures **global structural properties** of a shape.

In this project:

- 🟢 **Skeleton joints are treated as a 3D point cloud**
- 🚫 No graph connectivity or bone topology is used during ECT computation
- 🔄 ECT operates independently of the skeleton graph

This makes ECT complementary to graph-based models such as HD-GCN.

---

### 4. ⚙️ ECT Computation (High-Level)

For each skeleton sequence:

1. 🧍 Skeleton joints are treated as a 3D point cloud  
2. 🧭 The point cloud is projected onto a small set of predefined 3D directions  
3. 📊 For each direction, Euler characteristics are computed across multiple thresholds  
4. 🧠 The resulting ECT curves are mapped by an MLP into a fixed-dimensional feature vector  

ECT features are computed **in parallel** with HD-GCN features.

---

### 5. 🔬 Explored ECT Integration Strategies

#### 5.1 🔗 ECT-Guided Topological Edges
- Using ECT similarity to modify graph connections  
- Result: increased complexity, no consistent gain  

#### 5.2 🎛️ ECT-Based Gating
- Using ECT features to gate GCN activations  
- Result: unstable training  

#### 5.3 ⏱️ Multi-Scale Temporal ECT
- Computing ECT at different temporal resolutions  
- Result: high cost, limited benefit  

#### 5.4 🎓 Learnable ECT Parameters
- Making projection directions learnable  
- Result: prone to overfitting, no clear advantage  

---

### 6. 🧭 Direction Initialization and Regularization

An important empirical finding concerns ECT projection directions:

- ✅ A **small number of fixed, approximately orthogonal directions** is sufficient  
- ❌ Learnable directions do not improve performance  
- 🎯 Proper initialization is more important than learnability  

Directions are initialized using uniformly distributed vectors on the sphere and kept fixed during training.

---

### 7. 🏁 Final Design: ECT-Fusion

The most effective design is **ECT-fusion**:

- 🧠 HD-GCN captures local spatio-temporal relations  
- 🌐 ECT captures global topological structure  
- ➕ The two features are fused by simple feature addition  

This design is **simple, stable, and effective**.

---

### 8. 📊 Experimental Results  
**NTU RGB+D 60 — Cross-Subject**

| Method | HD-GCN Paper | This Project |
|------|--------------:|-------------:|
| Joint Stream | 90.4 | **90.57** |
| Bone Stream | 90.7 | **91.07** |
| Joint + Bone (Ensemble) | 92.4 | **92.61** |

---

### 9. 💡 Key Takeaways

- 🌍 ECT provides complementary global topology information  
- ⚖️ Complex ECT-based designs are not necessarily better  
- ⭐ Simple ECT-fusion is the most robust and effective strategy  
- 🧭 Fixed, well-initialized directions are sufficient for skeleton graphs  

---

### 10. 📝 Summary

This course project systematically explores **Euler Characteristic Transform (ECT)** for skeleton-based action recognition.

By evaluating multiple integration strategies, I show that **simple ECT-fusion achieves the best performance**, improving upon a strong HD-GCN baseline without increasing model complexity.
