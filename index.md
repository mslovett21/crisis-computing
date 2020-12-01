

# **CRISIS COMPUTING:** Multimodal Social Media Content for Improved Emergency Response 
<br/>

## **Motivation**

Multimodal data shared on Social Media during critical emergencies often contains useful information about a scale of the events, victims and infrastructure damage. This data can provide local authorities and humanitarian organizations with a big picture understanding of the emergency. Moreover, it can be used to effectively and timaly plan relief responses.
<br/>

![1f976786-3b50-429d-bf0c-d11ab5e85c6f](https://user-images.githubusercontent.com/31839963/100529409-4360ab00-319c-11eb-804e-223d0418ffb6.jpg)

<br/>

<span style="color:red">**Challenge 1:**</span> One of the biggest challenges is handling the social media information overload. To extract relevant information a computational system needs to process massive amounts of data and identify which data is <span style="color:blue">**INFORMATIVE**</span> in the context of disaster response.

![Firefox_Screenshot_2020-12-01T00-54-42 263Z](https://user-images.githubusercontent.com/7771314/100683240-c3b61600-332c-11eb-9afb-bf910d2aa8f9.png)

Subtle difference in visual characteristics of 2 images from **CrisisMMD** Dataset. Both images were published on Twitter between 27th August 2017 and 3rd September 2017.
<br/>

<span style="color:red">**Challenge 2:**</span> Another issue in the field is that of data scarsity. To develop effective applications that could assisit in crisis response, researchers need access to large-scale annotated dataset. 

In our work, we chose to explore current methodologies that can help alleviate these challenges. We decided to persue the following 2-fold problem statement.

## **Problem Statement**

- Explore different techniques of representation learning to improve performance on nuance classification of informative vs non-informative social media post in domain of crisis computing.
- Investigate applications of unsupervised and semi-supervised learning methods to mitigate the issue of labeled data scarcity on the classification task.


## **Methods**


<span style="color:blue"> **REPRESENTATION LEARNING: CONTRASTIVE LEARNING** </span> 

To improve classification accuracy on <span style="color:blue"> *informative* </span> vs. <span style="color:blue"> *non-informative*</span> classification task we decided to use supervised methods that produce meaningful, low-dimensional representations of the data.


<span style="color:red"> ***SupCon Architecture*** </span> 
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100697371-af811180-334a-11eb-9d1b-e5a72f5558f6.png" width="900" height="500">
</p>




<span style="color:orange"> **REPRESENTATION LEARNING: SENTENCE EMBEDDINGS** </span> 

<span style="color:red"> ***Fine-Tuned DistilBERT*** </span> 

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100696724-17365d00-3349-11eb-9168-d1322a456501.png" width="900" height="500">
</p>



<span style="color:orange"> **UN- and SEMI-SUPERVISED LEARNING** </span> 
<br/>
<span style="color:red"> ***FixMatch*** </span> 




![5c67944c-8d8f-488d-a67f-7bbc69d4e588](https://user-images.githubusercontent.com/31839963/100529666-af441300-319e-11eb-9035-4d3e0f0b70f9.jpg)


## **Experiments**


<span style="color:orange"> **Late Fusion** </span> 



**Modality** | **Model** | **Our Accuracy** | **Gautam et al. 2019**
------------ | --------- | ---------------- | ----------------------
**Text** | BiLSTM | **84.0** | 82
**Image** | ResNet50 | **88.1** | 79
**Text + Image** | BiLSTM + ResNet50 | **92.0** | 80


**Model** | **Mean Probability** | **Custom Decision** | **Logistic Regression**
------------ | --------- | ---------------- | ----------------------
**Our Model** (BiLSTM + ResNet50) | **92.0** | **91.8** | **91.5**
**Gautam et al.** (best among baselines) | 79.2 | 80.2 | 80.2


![7791bd18-f4ba-47e4-9dd8-fc5b6b602955](https://user-images.githubusercontent.com/31839963/100530749-90e41480-31aa-11eb-997b-ec18607f03bd.jpg)



<span style="color:orange"> **Early Fusion** </span> 


![pasted image 0](https://user-images.githubusercontent.com/31839963/100530769-cf79cf00-31aa-11eb-88d0-227b8481aa70.png) | ![f82882d9-5259-47c9-acd9-ca33d785a711](https://user-images.githubusercontent.com/31839963/100530624-11a21100-31a9-11eb-91e5-f5d4b4579c8c.jpg)
------------------------------------------------------------------------------------------------------------------------ | -----------------------------------------------------------------------------------------------------------------------------------------------
Olfi et al. architecture for the multimodal classification task | Our architecture for the multimodal classification task


**Accuracy**
**Modality** | **Olfi et al.** | **Ours**
------------ | --------------- | --------
**Text** | 0.808 | **0.84**
**Image** | 0.833 | **0.89**
**Text + Image** | 0.844 | **0.91**


**epoch** | **Training Loss** | **Valid. Loss** | **Valid. Accur.**
--------- | ----------------- | --------------- | -----------------
**1** | 0.59 | 0.47 | 0.80
**2** | 0.44 | 0.41 | 0.83
**3** | 0.35 | 0.40 | 0.83
**4** | 0.30 | 0.40 | 0.84


## ISSUE OF LABELED DATA SCARCITY


**Model** | **Modality** | **Accuracy**
--------- | ------------ | ------------
**kNN with SupCon** | **Image** | **75.9%**
**DeCLUTR** | **Text** | **73.4%**
**FixMatch** | **Image** | **In progress**


**#####ADD THE NEW PLOT FOR SUPCON IMAGE EMBEDDINGS#####**


## IMPROVING THE PERFORMANCE USING PSEUDO-LABELS


**#####FINAL RESULTS#####**



# RESULTS


**THE POWER OF MULTIMODAL DATA**


![be39ece1-d2f0-4e87-a95b-fde2bc2bce01](https://user-images.githubusercontent.com/31839963/100531114-36998280-31af-11eb-8fbf-be65227a7168.jpg)


## **Future Work**


- Improve methods/algorithms for obtaining enriched features from text and images
- Improve label propagation techniques
- Automatically generate a coherent summary report about an emergency event

<center>
![unnamed](https://user-images.githubusercontent.com/31839963/100531135-5a5cc880-31af-11eb-9a99-fe46c38032ea.png)
</center>







