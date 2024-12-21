# EDoF-CNN-Pack

This repository is forked from https://github.com/tomealbuquerque/EDoF-CNN

Acknowledgement to 
Tomé Albuquerque, Luís Rosado, Ricardo Cruz, Maria João M. Vasconcelos, Tiago Oliveira and Jaime S. Cardoso

https://doi.org/10.1016/j.iswa.2022.200170

_________________________________________________________________________________________________________________________

## Convolutional neural network with packing mechanism and color transfer post processing for extended depth of field cervical cytology images

by TANGPORNPISIT THANAWAT (Kanazawa University), HANDAYANI LILIES (Kanazawa University), CHEGODAEV DENIS (Kanazawa University), 
RAES RIK GIJS GERRIT (Kanazawa University), SATOU KENJI (Kanazawa University)

## Usage
  
  1. Run the aligment method (Chromatic/Rigid/elastic) if your dataset is misaligned.
  2. Run datasets\prepare_{dataset name}.py to generate the data.
  3. Run train.py to train the models you want.
  4. Run test.py to get the test result.
  5. Run evaluate.py to generate aggregate results.

## EDoF_CNN_Pack for RGB

Please train red, green, and blue channel separately. Then select rgb channel for test procedure.

## Post Processing

Please select post_processing True or False during test procedure to get the appropriate test result.