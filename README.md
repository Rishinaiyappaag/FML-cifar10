# Federated Learning on CIFAR-10 using Flower

This project implements Federated Learning on the CIFAR-10 dataset using the Flower framework and PyTorch. Multiple clients train a shared global model collaboratively without sharing raw data, preserving privacy.

---

## Overview

Federated Learning (FL) is a distributed machine learning approach where multiple clients train a model locally and send only model updates to a central server. The server aggregates these updates to improve the global model.

This project demonstrates:

- Federated training on CIFAR-10
- Client-server architecture using Flower
- Distributed model training and aggregation
- Performance tracking across communication rounds

---


---

## Technologies Used

- Python
- PyTorch
- Flower (Federated Learning Framework)
- CIFAR-10 Dataset
- NumPy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rishinaiyappaag/FML-cifar10.git
cd FML-cifar10
