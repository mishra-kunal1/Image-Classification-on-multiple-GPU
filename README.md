# Image-Classification-on-multiple-GPU

#The Dataset 
What is that person saying? American Sign Language is a natural language that serves as the predominant sign language of Deaf communities in the United States of America. ASL is a complete and organized visual language that is expressed by employing both manual and non manual features. However, with the advent of technology we can devise a more sophisticated solution which can translate this sign language into speech.
 
In this project we aim to do so with the help of a Deep Learning Model trained to classify the images. Also to enhance the model’s performance and it’s efficiency we are implementing this project using DataParallelism and DistributedDataParallelism techniques.
![image](https://user-images.githubusercontent.com/99056351/215806401-d5406a67-7429-44de-9c20-0afa6334cf30.png)

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
 
These 3 classes are very helpful in real-time applications, and classification. The test data set contains a mere 29 images, to encourage the use of real-world test images.
![image](https://user-images.githubusercontent.com/99056351/215806547-df9632f6-dd6f-43ba-886f-232b2ab9404e.png)

#Preparing the dataset 

The dataset used for this project consists of images which are required to be converted into a desirable computationally efficient format in order to be fed as input to our model. We have built a helper function to do so called prepare_dataset() as shown in the image below. 
 
As part of preprocessing the data we first apply image transformation to convert the input images into tensors and further normalization to remove anomalies and eliminate transitive dependency if any. And in order to efficiently train our model later we sample the data and define training and validation subsets.
![image](https://user-images.githubusercontent.com/99056351/215806727-32815714-438e-4989-8988-7b524709803a.png)


