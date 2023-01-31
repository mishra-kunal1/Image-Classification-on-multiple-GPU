# Image-Classification-on-multiple-GPU

## The Dataset 

What is that person saying? American Sign Language is a natural language that serves as the predominant sign language of Deaf communities in the United States of America. ASL is a complete and organized visual language that is expressed by employing both manual and non manual features. However, with the advent of technology we can devise a more sophisticated solution which can translate this sign language into speech.
 
In this project we aim to do so with the help of a Deep Learning Model trained to classify the images. Also to enhance the model’s performance and it’s efficiency we are implementing this project using DataParallelism and DistributedDataParallelism techniques.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/99056351/215807223-73927ea5-bead-4a96-8dbe-ab9d4bed5483.png">


The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
 
These 3 classes are very helpful in real-time applications, and classification. The test data set contains a mere 29 images, to encourage the use of real-world test images.
<img width="197" alt="image" src="https://user-images.githubusercontent.com/99056351/215807500-b7d0e6e9-cab2-4ea9-ad8a-b889d1bcab03.png">

## Preparing the dataset 

The dataset used for this project consists of images which are required to be converted into a desirable computationally efficient format in order to be fed as input to our model. We have built a helper function to do so called prepare_dataset() as shown in the image below. 
 
As part of preprocessing the data we first apply image transformation to convert the input images into tensors and further normalization to remove anomalies and eliminate transitive dependency if any. And in order to efficiently train our model later we sample the data and define training and validation subsets.

<img width="369" alt="image" src="https://user-images.githubusercontent.com/99056351/215807556-ab75d6b5-1848-42e2-8dca-037f2990cd7c.png">

## Building the Model
As discussed we have implemented 2 parallel approaches to train our model as with DataParallel and DistributedDataParallel. In both the cases, we have defined the model with the help of a helper function described in below sub-modules.
i) Using Pythorch’s DataParallel module
The model_loader() function takes the batch_size and data_subsets as arguments. The function first samples the data as per batch_size and loads them as per training and validation data loader.
 <img width="347" alt="image" src="https://user-images.githubusercontent.com/99056351/215808127-0958c7ee-0c31-41a1-b3fe-730263f962c6.png">

Then we initialize our model and convert the model using torch’s DataParallel class by passing the model as the object resnet = nn.DataParallel(resnet) Adn. Once the model has been instantiated we modify the model’s fully connected(fc) layer in order to yield output into the required number of categories, in this case the length of the classes variable as seen from the code. 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/99056351/215808182-c810c2da-9651-461a-a80d-645a3e018d34.png">

ii) Using Pythorch’s DataDistributedParallel module
Before using the data preprocessed from the previous step we first Data Parallelism using the DataLoader() function from torch. By adjusting num_workers parameter we can define in how many partitions the data should be divided and then each partition will be given to a single GPU at a time for processing. After defining the model we implement Model Parallelism using the DistributedDataParallel class from torch.parallel module. Once the model is initialized we re-define the final fully-connected(fc) layer of the model as per our requirement. We can also define different models and modify their final layers as per their architecture.
 
## Training the Model
Now, we are required to train the model. And the training process for every model varies subjectively. First we define the training procedure for DataParallel Mechanism as follows.
i) Training of model using DataParallel module (for 10k samples)
We have defined the train() helper function in order to train the model. We receive all the hyper parameters and arguments to train the model from the calling main function. The training procedure is simple and straightforward. We also record time taken each time whenever we increase the number of GPUs per session to run our code and compare the model’s performance. 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/99056351/215808323-4b94a79f-c10b-4dda-80f1-6e5a45952fe8.png">

The detailed analysis of the time taken under different numbers of GPUs is illustrated in the below chart with resnet Model.
<img width="565" alt="image" src="https://user-images.githubusercontent.com/99056351/215808462-c3286d90-54a5-4925-b2c9-becb5d42fd01.png">

