# Destination Choice Modelling with Perceptual Features

This repository contains Python code developed for a Final Year Project (FYP) on destination choice modelling. The project integrates behavioural modelling, spatial data processing, and computer vision–based perceptual feature extraction.

## Overview

The study examines shopping and leisure destination choice by incorporating both traditional travel impedance variables and street-level perceptual attributes, including:

- Vibrancy  
- Pleasantness  
- Walkability  
- Safety  
- Experiential richness  

A Hierarchical Bayesian Hybrid Choice Model is implemented, combining:

- Stated Preference (SP) survey data  
- Origin–Destination (OD) flow data  
- Computer Vision–derived environmental features  

## Repository Structure

The repository is organized following the Appendix structure of the thesis to ensure full reproducibility of the analytical pipeline:
- Appendix_A_Spatial_Framework/
- Appendix_B_Origin_Table/
- Appendix_C_Destination_Table/
- Appendix_D_Travel_Skims/
- Appendix_E_SP_Survey/
- Appendix_F_Model/
- Appendix_G_Robustness_Checks/
- README.md

### Description of Each Module
### Description of Each Module

- **Appendix A – Spatial Framework, Data Sources, and Study Scope**  
  Construction of the spatial analysis framework, study boundaries, and integration of raw data sources.

- **Appendix B – Origin Table Construction**  
  Processing and aggregation of origin-level data.

- **Appendix C – Destination Table Construction**  
  Assembly of destination (POI) database and extraction of perceptual features using computer vision.

- **Appendix D – Travel Skims Calculation**  
  Computation of travel impedance matrices, including travel time, cost, and transfers.

- **Appendix E – SP Survey Design**  
  Experimental design and generation of stated preference choice scenarios.

- **Appendix F – Model Specification and Estimation Results**  
  Implementation of the Hybrid Choice Model and estimation procedures.

- **Appendix G – Robustness Checks**  
  Sensitivity analysis and robustness validation of model results.

## Requirements

Main dependencies include:

- Python 3.x  
- numpy  
- pandas  
- geopandas  
- scikit-learn  
- pymc / arviz  
- matplotlib  

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Workflow

The codebase follows a sequential pipeline:
	1.	Define spatial framework and integrate data (Appendix A)
	2.	Construct origin and destination datasets (Appendix B–C)
	3.	Compute travel skims (Appendix D)
	4.	Design SP experiment (Appendix E)
	5.	Estimate models (Appendix F)
	6.	Conduct robustness checks (Appendix G)

## Author

Zhang Wenyu

## License

This project is intended for academic use. Please contact the author for reuse or collaboration.
