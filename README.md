Segmentation of the aortic root using MO during transcatheter aortic valve implantation

📖 Contents
Introduction
Data
Methods
Results
Conclusion
Requirements
Installation
How to Run
Data Access
How to Cite

🎯 Introduction
This repository presents an artificial intelligence (AI)-driven approach for the precise segmentation and quantification of histological features observed during the microscopic examination of tissue-engineered vascular grafts (TEVGs). The development of next-generation TEVGs is a leading trend in translational medicine, offering minimally invasive surgical interventions and reducing the long-term risk of device failure. However, the analysis of regenerated tissue architecture poses challenges, necessitating AI-assisted tools for accurate histological evaluation.

📁 Data
The dataset used in the study included 2854 contrast-enhanced images of 1000 × 1000 pixels in size with a color depth of 8 bits (scale from 0 to 255). The final sample consisted of 2854 grayscale images, of which 2455∽2514 images (∽87%) or 65 patients constituted the training set, and 340∽399 (∽13%) images or 15 patients were used as the validation set. As part of the performed TAVI procedures, a series of anonymized images were acquired illustrating four main steps: survey angiography (Figure 1A), positioning of the catheter and delivery system (Figure 1B); initiation of retraction of the delivery system and valve exposure (Figure 1C); control angiography after valve implantation (Figure 1D).

![image](https://github.com/user-attachments/assets/7b6db0bc-c903-40e0-bece-bbc7d8f74a1c)
Figure 1. Data for labeling intraoperative aortography images during the TAVI procedure.

📈 Results
DeepLabV3+ and U-Net++ achieved the best DSC values ​​of 0.877–0.881 and had the smallest range of DSC values, which is the most preferred option for clinical trials.
MA-Net and LinkNet also showed good results while requiring less computational resources, which is suitable for resource-constrained systems.
FPN requires additional optimization to reduce the range of DSC values, despite the maximum DSC values ​​of 0.916.
PSPNet is not recommended for tasks that are critical for stability and convergence speed.


🏁 Conclusion
This study highlights the potential of deep learning models for accurate segmentation of the aortic root, paving the way for optimized workflows utilizing AI during the TAVI procedure. The results contribute to further research in this field and foster the development of intelligent visual assistants.

💻 Requirements
Operating System
 macOS
 Linux
 Windows (limited testing carried out)
Python 3.11.x
Required core libraries: environment.yaml

⚙ Installation
Step 1: Install Miniconda

Installation guide: https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

Step 2: Clone the repository and change the current working directory

git clone https://github.com/ViacheslavDanilov/histology_segmentation.git
cd histology_segmentation
Step 3: Set up an environment and install the necessary packages

chmod +x make_env.sh
./make_env.sh

🔐 Data Access
All essential components of the study, including the curated source code, dataset, and trained models, are publicly available:

Source code: https://github.com/Nikita75699/segmentation_tavi.git
Dataset: https://doi.org/10.5281/zenodo.15094600
Models: https://zenodo.org/10.5281/zenodo.15094680
