# Gait Analysis with Graph Signal Processing

This repository contains part of the code used in my **Master’s Thesis**:

> **Graph Fourier Transform: A Study of Time-Dependent Graph Signals**  
> University of Padua, 2025  
> [Full thesis available here](https://thesis.unipd.it/handle/20.500.12608/91831)

The project explores **Graph Signal Processing (GSP)** techniques applied to human gait analysis.  
Specifically, it models human joint dynamics as signals evolving over a 3D skeletal graph,  
analyzing motion through spectral methods such as the **Graph Fourier Transform (GFT)**.

---

## 📂 Repository Structure
```
.
├── config/       # Graph and gait configurations
├── src/          # Main source code (GraphModel, GraphSignal, GaitTrial, Visualizer, etc.)
├── notebooks/    # Example notebooks demonstrating object usage and workflow
├── results/      # Generated plots, animations, and analysis results
└── README.md
```


---

## 🧠 Project Description

This repository provides object-oriented tools for representing and analyzing  
**time-varying graph signals** derived from human motion data.  
It includes:

- Construction of the **graph structure** (joints and bone connections)  
- Definition of **graph signals** for position and velocity  
- Implementation of **spectral filtering** in the GFT domain  
- Visualization utilities for eigenmodes, GFT energy evolution, and 3D gait animations

---

## 💾 Dataset

The dataset used in this work is **not included** due to size constraints.  
It can be retrieved from:

> [Azure Kinect 3D skeleton and foot pressure data for pathological gaits](https://ieee-dataport.org/documents/azure-kinect-3d-skeleton-and-foot-pressure-data-pathological-gaits)

Once downloaded, adjust your local paths inside the project to point to the dataset’s root folder.

---

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rafael-dimare/gft_gait_analysis.git
   cd gft_gait_analysis
