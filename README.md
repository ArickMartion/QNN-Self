# QNN-Self

# Quantum Neural Network for Nonlinear Data Classification

A custom-built Quantum Neural Network (QNN) compatible with photonic quantum chips, designed for solving nonlinear classification problems using data re-encoding strategies.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MindQuantum](https://img.shields.io/badge/MindQuantum-0.11.0-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## 📋 Overview

This project implements a fully customizable QNN architecture using MindQuantum framework, specifically designed for:
- **Photonics Quantum Chip Compatibility**: All components are tailored for synchronization with experimental photonic quantum hardware
- **Nonlinear Classification**: Solving custom ring-shaped nonlinear data classification problems
- **Custom Gradient Control**: Manual implementation of forward propagation, gradient computation, and parameter updates
- **Data Re-encoding**: Advanced data encoding strategies for quantum advantage

## 🎯 Features

- **Custom Quantum Circuit Design**: Flexible circuit construction for photonic quantum processors
- **Manual Gradient Computation**: Full control over quantum gradients for hardware synchronization
- **Data Re-encoding Pipeline**: Quantum data encoding strategies for nonlinear problems
- **Visualization Tools**: Comprehensive plotting utilities for quantum states and training progress
- **Modular Architecture**: Clean separation of QNN components for easy experimentation

## 📁 Project Structure
├── My_QNN.py # Custom QNN class definition

├── My_QCircuit.py # Custom quantum circuit construction

├── My_QClassification_Algorithm.py # Classification algorithms and utilities

├── My_plot.py # Custom plotting functions

├── Learn Decision_Boundary-Circle-Parameter_Shift.ipynb  # Parameter-shift rule for exact gradient computation (high precision but slower)

├── Learn Decision_Boundary-Circle-SPSA.ipynb            # SPSA for stochastic gradient approximation (faster but approximate)

└── README.md # This file
