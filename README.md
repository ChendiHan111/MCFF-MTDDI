# MCFF-MTDDI
This is the code for our paper "MCFF-MTDDI: Multi-Channel Feature Fusion for Multi-Typed Drug-Drug Interaction Prediction".
## 1 System requirements:
```
Hardware requirements:
        multi-class-prediction1.py, multi-label-prediction1.py, multi-class-prediction2.py and multi-label-prediction2.py require a computer with enough RAM to support the in-memory operations.
        Operating system：windows 10
        
Code dependencies:
        python '3.8' (conda install python==3.8)
        pytorch-GPU '1.9.1' (conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio== 0.9.1 cudatoolkit=11.1 -c pytorch)
        numpy '1.20.3' (conda install numpy==1.20.3)
        pandas '1.3.4' (conda install pandas==1.3.4)
        scikit-learn ' 1.0.1' (conda install scikit-learn==1.0.1)
```
## 2 Installation guide:
```
First, install CUDA 11.1 and CUDNN 8.0.5.
Second, install Anaconda3. Please refer to https://www.anaconda.com/distribution/ to install Anaconda3.
Third, install PyCharm. Please refer to https://www.jetbrains.com/pycharm/download/#section=windows.
Fourth, open Anaconda Prompt to create a virtual environment by the following command:
	conda env create -n env_name python=3.8
```
## 3 Instructions for use(Four benchmark datasets are included in our data):
```
Based on Multi-Class DDI dataset1:
        First, put folder Multi-Class-data1, folder input-features and multi-class-prediction1.py into the same folder.
        Second, use PyCharm to open multi-class-prediction1.py and set the python interpreter of PyCharm.
	Third, modify codes in multi-class-prediction1.py to set the path for loading data and the path for saving the trained model.
	Fourth, multi-class-prediction1.py in PyCharm.
        
        Expected output：
		A txt file with timestamps and results of all evaluation metrics.
    
Based on Multi-Label DDI dataset1:
        First, put folder Multi-Label-data1, folder input-features and multi-label-prediction1.py into the same folder.
        Second, use PyCharm to open multi-label-prediction1.py and set the python interpreter of PyCharm.
	Third, modify codes in multi-label-prediction1.py to set the path for loading data and the path for saving the trained model.
        Fourth, multi-label-prediction1.py in PyCharm.
        
        Expected output：
		A txt file with timestamps and results of all evaluation metrics.
                
Based on Multi-Class DDI dataset2:
        First, put folder Multi-Class-data2, folder input-features and multi-class-prediction2.py into the same folder.
        Second, use PyCharm to open multi-class-prediction2.py and set the python interpreter of PyCharm.
	Third, modify codes in multi-class-prediction1.py to set the task for prediction, the path for loading data and the path for saving the trained model.
	Fourth, multi-class-prediction2.py in PyCharm.
        
        Expected output：
		A txt file with timestamps and results of all evaluation metrics.  
                
Based on Multi-Label DDI dataset2:
        First, put folder Multi-Label-data2, folder input-features and multi-label-prediction2.py into the same folder.
        Second, use PyCharm to open multi-label-prediction2.py and set the python interpreter of PyCharm.
	Third, modify codes in multi-label-prediction2.py to set the task for prediction, the path for loading data and the path for saving the trained model.
	Fourth, multi-label-prediction2.py in PyCharm.
        
        Expected output：
		A txt file with timestamps and results of all evaluation metrics.                
```
