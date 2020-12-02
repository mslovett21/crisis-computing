

# **CRISIS COMPUTING:** Multimodal Social Media Content for Improved Emergency Response 
<br/>

## **Motivation**

Multimodal data shared on Social Media during critical emergencies often contains useful information about a scale of the events, victims and infrastructure damage. This data can provide local authorities and humanitarian organizations with a big picture understanding of the emergency. Moreover, it can be used to effectively and timely plan relief responses.
<br/>

![1f976786-3b50-429d-bf0c-d11ab5e85c6f](https://user-images.githubusercontent.com/31839963/100529409-4360ab00-319c-11eb-804e-223d0418ffb6.jpg)

<br/>

<span style="color:red">**Challenge 1:**</span> One of the biggest challenges is handling the social media information overload. To extract relevant information a computational system needs to process massive amounts of data and identify which data is <span style="color:blue">**INFORMATIVE**</span> in the context of disaster response.

![Firefox_Screenshot_2020-12-01T00-54-42 263Z](https://user-images.githubusercontent.com/7771314/100683240-c3b61600-332c-11eb-9afb-bf910d2aa8f9.png)

Subtle difference in visual characteristics of 2 images from **CrisisMMD** Dataset. Both images were published on Twitter between 27th August 2017 and 3rd September 2017.
<br/>

<span style="color:red">**Challenge 2:**</span> Another issue in the field is that of data scarcity. To develop effective applications that could assisit in crisis response, researchers need access to large-scale annotated dataset. 

In our work, we chose to explore current methodologies that can help alleviate these challenges. We decided to pursue the following 2-fold problem statement.

## **Problem Statement**

- Explore different techniques of representation learning to improve performance on nuance classification of informative vs non-informative social media post in domain of crisis computing.
- Investigate applications of unsupervised and semi-supervised learning methods to mitigate the issue of labeled data scarcity on the classification task.


## **Methods**


<span style="color:blue"> **REPRESENTATION LEARNING: CONTRASTIVE LEARNING** </span> 

To improve classification accuracy on <span style="color:blue"> *informative* </span> vs. <span style="color:blue"> *non-informative*</span> classification task we decided to use supervised methods that produce meaningful, low-dimensional representations of the data.

<span style="color:orange"> ***SupCon Overview*** </span>

For our images, we use contrastive learning paradigm that is well-suited for embedding nuance concepts.
We aim to learn good representations of the 2 classes so that the downstream classification task becomes easy. Contrastive learning in a nutshell tries to pull clusters of points belonging to the same class close together in the embedding space, while simultaneously pushing apart the clusters of samples from different classes.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100697168-1651fb00-334a-11eb-9c42-db9e35777c37.png" width="900" height="400">
</p>

There are many unsupervised methods for this type of problems but we chose to use the Supervised Contrastive Learning algorithm where we leveraged the labelled information due to the nature of our classification task. Once the model is trained, we use it to the created embeddings for a downstream tasks, as well as, for a never-seen subset of the data and check the accuracy of the model through kNN.   <br/>

How SupCon works?
  1. Given an input batch of data, we first apply data augmentation twice to obtain two copies of the same batch. Both the copies are forward propagated through the encoder network to obtain a 2048-dimensional normalized embedding. The network learns about these transformations, what it means to come from the same image, how to spread data in embedding space, etc.
  2. During training, this representation is further propagated through a projection network which is discarded at inference time. The supervised contrastive loss is then computed on the outputs of the projection network
  3. Can use linear classifier or KNN to predict the classes of new examples. The contrastive loss maximizes the dot products of embeddings from similar classes and separates the positive samples from negatives using labels to make the distinction.
<br/>


<span style="color:blue"> **REPRESENTATION LEARNING: SENTENCE EMBEDDINGS** </span> 

<span style="color:orange"> ***Fine-Tuned DistilBERT*** </span>
<br/>
To create sentence embeddings that perform well in the crisis computing context, we decided to fine-tune DistilBERT model (from huggingface) on our downstream task of tweets classification. The modele is pretrained on the GloVe Twitter 27 B embeddings. Our tweets are first preprocessed; we remove stop words, URLs,hastags and punctuation. 

<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100827860-993e8880-3412-11eb-874d-6a5792613737.png" width="900" height="200">
</p>
<br/>
Next, we use DistiBERT tokenizer. We trained the model for 4 epochs with batch size of 16. We use Adam optimizer with weight decay of 0.01 and custom weighted loss function that compensates for the unbalanced dataset.

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100704075-b06d6f80-3359-11eb-92b8-e39b361a8004.png" width="900" height="500">
</p>
We extrace sentence embeddings from the fine-tunes model by avaraging all the final hidden layers of all the tokens in the sentences. The latent space vectors are extracted for the training data, as well as, for test data that the model have not seen before. We use the embeddings during in our early fusion architecture.

<br/>

<span style="color:blue"> **SEMI-SUPERVISED LEARNING** </span> 
<br/>
<br/>
<span style="color:orange"> ***FixMatch*** </span>

To address the issue of data scarcity, we use semi-supervised learning in the form of consistency regularization and pseudo-labelling.
Our goal is to provide a label to the unlabeled images in our dataset in order to obtain large-scale annotated dataset for developing effective applications.
<br/>


<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100698174-aee97a80-334c-11eb-9839-f85bd999ce0f.png" width="800" height="300">
</p>

FixMatch uses both the approaches together to generate highly accurate labels, by following a two-step method:
  1. Creates a weakly-augmented version of the unlabeled image using basic augmentations like flip-and-shift, and produces a pseudo-label for it using the model's predictions, which is retained only if it's confidence is above a specific threshold.
  2. Feeds a strongly-augmented version of the same image to the model and trains it with pseudo-label as the target using cross-entropy loss.
<p>
FixMatch requires extensive GPU utilization and we aim to obtain higher accuracy with better GPUs available.
</p>

<br/>
<br/>

<span style="color:blue"> **MULTIMODAL LEARNING** </span> 
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100710225-43aba280-3364-11eb-97d6-41c1b38078e9.png">
  </p>
Information from a single source is good but wouldn't it be better to get additional information from multiple sources? Exactly! There are multiple sources of data for a single problem at hand. These sources offer complementary information which not only helps to improve the performance of the model but also enables the model to learn better feature representations by utilizing the strengths of individual modalities.
For instance, visual information from images is very sparse, whereas a piece of textual information for the same is more expressive. Combining these two gives us enriched information about the scene at hand. We attempt to employ this intuition by exploring early and late fusion techniques to achieve robust performance.
 <br/>
 <br/> 

<span style="color:orange"> **The Power of Multimodal Data** </span> 

<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/31839963/100531114-36998280-31af-11eb-8fbf-be65227a7168.jpg" width="700" height="400">
</p>
[Image source: Olfi et al. 2020]
  <br/>

## **Experiments**


<span style="color:blue"> **MULTIMODAL LEARNING: Late Fusion** </span> 
<br/>
To handle the modalities of the dataset, we combine the representations of text and image by performing three late fusion techniques. We combine our best models Bi-LSTM for text and ResNet-50 for images and compare the results with the best baselines from Gautam et al.

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100700177-b7907f80-3351-11eb-830f-ab8c4930a67e.png" width="600" height="200">
</p>
<br/>
Our improved model was able to beat the baselines with a huge margin in all three fusion techniques viz. Mean Probability Concatenation, Custom Decision Policy and Logistic Regression Decision Policy.

- We saw improved performances in all these techniques because of efficient base models i.e. ResNet-50 for image modality and Bi-LSTM for text modality with better accuracies than the baselines.

- In case of Custom Decision Policy, we implemented  2 fully connected layers with ReLU activaton function in the first 128 dim layer and Sigmoid in the last layer. We trained the model for 30 epochs with Adam optimizer and BCE Loss to obtain a 0.08 training-loss.
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100701939-7ef2a500-3355-11eb-8b44-8c30cd272853.png" width="900" height="500">
</p>
<br/>
From the above confusion matrices it is quite evident that our model has a higher AUCROC value since it is able to distinguish between the two classes effectively. The model identifies a large amount of true positives and true negativies thereby making it a robust for the task. The accuracy of these techniques can be further increased with better and deeper base architectures for text and image modalities.
<br/>
<br/>

<span style="color:blue"> **MULTIMODAL LEARNING: Early Fusion** </span> 
<br/>
We take our work step futher, 
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100702240-37204d80-3356-11eb-88b6-d38335ff1bc7.png" width="500" height="300">
</p>
<br/>

<br/>


<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100702866-4eac0600-3357-11eb-80a9-1d53e7989498.png" width="1000" height="350">
</p>
<br/>

<br/>

**Modality** | **Olfi et al.** | **Ours**
------------ | --------------- | --------
**Text** | 0.808 | **0.84**
**Image** | 0.833 | **0.89**
**Text + Image** | 0.844 | **0.91**


## **Label Data Scarcity**
## Improve the Performance using Pseudo-Labels

<br/>

**Model** | **Modality** | **Accuracy**
--------- | ------------ | ------------
**kNN with SupCon** | **Image** | **75.9%**
**DistilBERT** | **Text** | **73.4%**
**FixMatch** | **Image** | **72.9%**


<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100822858-013ba180-3408-11eb-80b5-6efc2e728b08.gif">
</p>
<br/>
<p>
The above scatter plot depicts one of the best results we have got so far after training our Supervised Contrastive model. We can see that our Supervised Contrastive learning model has done a great job at segregating the informative and non-informative images. However, there have been some scenarios where our network fails to classify the images. This is mainly due to the graphical images which depict the statistical information of the disaster. For eg: It was been observed that all the Informative graphical image training data showed a biasness with respect to the line graph meaning most of the graphical informative images were that of a line graph and hence our model had a lot of false positives when tested on unseen data (eg. random images downloaded from google.com) which involved random line plot graphs. 
Hence our future scope of the study would be to modify our model in such a way that it would predict such kind of graphical images depending upon the graph plot information passed along with the graph, which is usually found at the top right region of the graph or with respect to the x-axis & y-axis label information of the graph.
</p>

<br/>

## **Future Directions**

<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100709069-4ad1b100-3362-11eb-9f1b-c507188c3ff1.png" width="700" height="300">
</p>
<br/>

Improve methods/algorithms for obtaining enriched features from text and images. Improve label propagation techniques. Automatically generate a coherent summary report about an emergency event.

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100708777-c41cd400-3361-11eb-9467-11632f404f54.png">
</p>





