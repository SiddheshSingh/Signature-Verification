# Signature-Verification
Offline Handwritten Signatures is the basic and most used form of Identification/Verification. This requires high manual work, which is in this case often not reliable as it depends on the person. So there is a requirement for a classifier which compares a real signature and an unknown signature, and determines correctly if the unknown signature is Real or Forged.

## Dataset
Total of 3 different datasets are used in this project. They can be downloaded from https://drive.google.com/drive/folders/1Wx3M8wCYKIXNzIK7QYg7azDppQz01Lvh?usp=sharing . <br><br>
**Signature Set 1** contains signatures (Genuine and Forged samples) of 30 writers, 5 sample each. <br>
This sample is used for final testing. <br><br>
**Signature Set 2** contains signatures (Genuine and Forged samples) of 55 writers, 24 sample each. <br>
This sample is used for training and validation/testing along with Signature Set 3.<br><br>
**Signature Set 3 (BHSig260)** is the main dataset used for training and validation/testing.

## Model
This project uses Siamese-Style Model. (PS, Not a Siamese Network using Triplet Loss Function, neither Contrastive Loss Function).
This is an attempt to use end to end deep learning for Signature Verification (One Shot Learning). <br> 
Therefore,
1) An **Embedding Model** is created. This, in theory, creates a vector representation (embeddings) of our images. <br>
2) Two of these Embedding Models are **Concatenated** (1st outputs the vector representation of 1st image, and 2nd outputs vector representation of 2nd image), and this concatenation is in turn connected to a Deep Neural Network. <br>
3) The **Deep Neural Network** interprets the embeddings and predicts whether the probability of the 2nd image being a real signature.

### Embedding Model
This model inputs an image, followed by convolutional layers and outputs the embeddings (Vector Representation of the image features). <br> <br>
Often, image-nets start with convolution layers, connecting to Dense Layers in the end. But in this model, *rather than using Dense Layer in the end of the network, **the Convolutional Neural Network is reduced to 1 x 1 with depth of 512, in turn, behaving itself like a Dense Layer.***<br> <br> 
**Input  :** 150 x 150 x 1 image.<br>
**Layer1 :** CNN(148, 148, 64) -> LeakyRelu -> BatchNorm -> MaxPool2D(2,2)<br>
**Layer2 :** CNN(72, 72, 128) -> LeakyRelu -> BatchNorm -> MaxPool2D(2,2) -> Dropout(0.2) <br>
**Layer3 :** CNN(34, 34, 256) -> LeakyRelu -> BatchNorm -> MaxPool2D(4,4) -> Dropout(0.2) <br>
**Layer4 :** CNN(6, 6, 512) -> LeakyRelu -> BatchNorm -> MaxPool2D(4,4) -> Dropout(0.2) <br>
**Layer5 :** CNN(1, 1, 512) -> LeakyRelu <br>
**Output :** Flatten(512)

### Final Network
This is the main network, which takes inputs, pass them through the embedding model, and processes the vector outputs from the models and outputs the probability of the 2nd image being real.<br>
*PS: Since there are 2 images inputted at once, Embedding Model is used twice in this case, but this doesn't mean there are twice as more parameters to train. We use the same model 2 times, so we only have to train the parameters once.*<br><br>
**Input  :** [ Input1-Layer(150,150,1) , Input2-Layer(150,150,1)] <br>
**Layer1 :** [Embedding-Model 1 , Embedding-Model 2] (Connected to Input1-Layer and Input2-Layer respectively)
**Concatenate** (1024) <br>
**Layer2 :** Dense(512) -> Relu -> Kernel Regularizers L2<br>
**Layer3 :** Dense(64) -> Relu <br>
**Output :** Dense(1) -> Sigmoid

#### Learning Rate and Optimizer
A Learning Rate Scheduler : **Exponential Decay** is used along with **RMS Prop**. <br>
Since a very big dataset is used, learning becomes a very slow process. Hence, initially a big learning rate is chosen, which later decays to a smaller value for proper decrease of the loss function.

## Results
#### Training Set Accuracy
The maximum training set accuracy is seen to be on 2nd epoch: **76.83%**.

#### Testing Set Accuracy
This model is tested upon 2 types of data.
* Data Similar to the training set. (Proper testing data)
* Data totally new. (The model is trained to predict Hindi Signatures better, so this is a test with the data in English Language)
##### Accuracy on Proper Testing Data
This dataset contains samples of unseen 16,000 signature pairs. Accuracy Achieved is **72%**
##### Accuracy on New Dataset
This dataset contains 600 pairs of English Signature pairs. Accuracy Achieved is **57.5%**

## How to run the code:
1) 3 datasets are used in this project available at https://drive.google.com/drive/folders/1Wx3M8wCYKIXNzIK7QYg7azDppQz01Lvh?usp=sharing , move the whole file in your google drive. <br>
2) Upload the BHSig260.zip in the colab notebook's environment. This speeds up the time as this file is very big, it would take hours to transfer from drive to colab. <br>
3) This code requires importing of data_processing.py. Either upload data_processing.py in colab notebook's environment, or else replace the line **import data_processing** in the cell 3 by the following lines: <br> *os.chdir('drive/My Drive/Signature_Verification')* <br>
*import data_processing* <br>
*os.chdir("/content")* <br>


