## Citation
﻿
If you use this code or data in your research, please cite our paper:
﻿
**Enhanced Structural Anomaly Detection through Improved Image Inpainting and Feature-Level Discrimination**  
Authors: Fei Dongce, Wu Qiling  
Journal: The Visual Computer  
Year: [Publication Year]  
DOI: [DOI of the Article]

## abstract

In industrial production, the detection of structural anomalies on product surfaces is crucial for quality control. Traditional encoder-decoder architectures for unsupervised anomaly detection, while effective, tend to reduce the reconstruction error between normal and abnormal samples due to their generalization capabilities. To address this, we propose an enhanced structural anomaly detection algorithm based on improved image inpainting. By converting the reconstruction task into an inpainting-filling-reconstruction process, we amplify the reconstruction error between normal and abnormal samples. Our algorithm uses a feature loss and feature-level anomaly discrimination method to mitigate noise interference. Furthermore, we introduce a lightweight U-Net design to meet industrial requirements. Experimental results on the MVTec LOCO AD dataset demonstrate that our approach outperforms existing algorithms, achieving an average AUROC of 85.0% for anomaly detection and 88.5% for anomaly localization, highlighting its potential in industrial applications.

## 1.Requirements

First, we need to split the MVTec LOCO AD dataset into two separate datasets: one for structural anomalies and the other for logical anomalies. The method for splitting is as follows:  
Primary Directory:  
![image](https://github.com/user-attachments/assets/7aae8bc3-a9de-4a22-9d47-691e335a7711)  
Second-level Directory:  
![image](https://github.com/user-attachments/assets/bfc9ca43-23dd-400e-b0c6-c152d1b09e20)  
Third-level Directory:  
![image](https://github.com/user-attachments/assets/fa6d9fef-d5f7-4546-bc5a-21fe00867b90)  
## 2. Proposed Method
This algorithm is based on an improved U-Net for image repair and reconstruction. The network is trained using an unsupervised learning method on a dataset without anomalies and requires two lightweight U-Nets. First, images are randomly masked using a mask to remove certain pixel information; then, the first lightweight U-Net is used to repair the masked regions. Next, the information generated by the first lightweight U-Net for the masked region is filled into the obscured sub-image, followed by reconstruction of the filled image using the second lightweight U-Net. The testing process differs from the training process, as the filled information generated by the first lightweight U-Net is used to fill the input image. The filled input image contains both real information and information generated by the network, which have consistent structural features, allowing the reconstruction network to reconstruct the normal regions with relative ease.  
Framework:  

![image](https://github.com/user-attachments/assets/dd7205d3-5429-4db3-bdc2-3902383850f9)  
The encoder and decoder each consist of five layers. The first layer includes a convolutional layer (conv), a batch normalization layer (BatchNorm), and an activation layer (ReLU). The remaining layers are composed of feature extraction modules. The structure of the feature extraction module is relatively straightforward, consisting of a group convolution, a depthwise convolution, and a standard   convolution.  
Network structure:  
![image](https://github.com/user-attachments/assets/38dd8db5-2c39-4b64-aba7-dee4ed881adb)  
Structural diagram of the feature extraction block:  
![image](https://github.com/user-attachments/assets/5d3a7c64-1eeb-40a7-82e2-265fbfadd28c)  
## 3. Key Algorithms
During the training and testing processes, we use different padding methods. When the parameter ‘select’ is set to 0, it indicates the training process; when ‘select’ is set to 1, it indicates the testing process. The specific implementation is as follows:
```
  if selecttest == 0:
      mb_cutout = mb_cutout + pre * (1 - mask)
  else:
      mb_cutout = mb_cutout + 0.5 * pre * (1 - mask) + 0.5 * mb_img * (1 - mask)
```
## 4. Code details
   (1) The main function is **train.py**.
   (2) **option.py** allows for the configuration of program parameters. The following two images represent the selection of datasets and the selection of data classes respectively.

   
Dataset Selection:```self.parser.add_argument('--dataroot', default='D:/fdc/chin\mvtec_loco_anomaly_detection_logical', help='path to dataset')``` 

Selection of Data classes:```self.parser.add_argument('--abnormal_class', default='juice_bottle', help='Anomaly class idx for datasets')``` 
## 5. result
Image level AUROC of different anomaly detection algorithms on the MVTec LOCO AD
|Class|GCAD|ComAD|RIAD|DADF|ESAD|
|-|-|-|-|-|-|
|Breakfast box|80.9|70.0|60.4|-|75.9|
|Screw| bag|70.5|	65.0|	81.8|	-|87.6|
|Pushpins|74.9|93.8|86.2|-|96.6|
|Juice bottle|98.9|80.5|96.8|-|97.8|
|Splicing connectors|78.3|63.8|94.6|-|88.5|
|Average str|80.7|74.6|84.0|88.2|89.3|

Display of images:  
![image](https://github.com/user-attachments/assets/0b74e3df-9bd1-45df-b2de-4873fb30f01c)

(a) Input sample; (b) Reconstructed image; (c) Label; (d) Anomaly prediction map generated by ESAD; (e) Anomaly prediction map overlaid on the original image


