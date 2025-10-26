# **Classify Alzheimer’s disease of the ADNI brain data using GFNet**
**Author**: Wenyue Guo (s4908581)  
**Project Number**: 8  
**Difficulty level**: Hard  
**Course**: COMP3710 - Pattern Analysis  
**Submission Date**: 30 October 2025  
**Version**: v1.0.0  
**License**: Apache License 2.0  
**License URL**: https://www.apache.org/licenses/LICENSE-2.0  
  
**Keyword**: Alzheimer's Disease, ADNI Dataset, GFNet, Medical Image Classification, Deep Learning  

### **Project Overview**

### **Project Structure**
GFNet_ADNI_s4908581/
    |--- modules.py 
    |--- dataset.py
    |--- train.py
    |--- predict.py 
    |--- README.md
### **Model Architecture**


### **Data Loading and Preprocessing**


### **Training and Testing**
#### **Training Result**

#### **Testing Result**

### **Design choices**
#### **Model Architecture**
I tried to apply residual connection to an individual block in order to increase stability.
The below table shows the comparison of two models
| | Block without residual connection | Block with residual connection |
|-------|-------|-------|
| Test Accuracy | 75.30% | 74.37 |
The below figures show the comparison of two architectures in the training process

Result shows that the stability of the two models is almost the same, but the test accuracy of block without residual connection is higher than the other one.

#### **Loss function**

#### **Optimizer**

#### **Scheduler**



### **Dependencies**


### **References**
