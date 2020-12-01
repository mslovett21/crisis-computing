

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

<span style="color:orange"> ***SupCon Architecture*** </span> 
text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100697168-1651fb00-334a-11eb-9c42-db9e35777c37.png" width="900" height="400">
</p>

text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>


<span style="color:blue"> **REPRESENTATION LEARNING: SENTENCE EMBEDDINGS** </span> 

<span style="color:orange"> ***Fine-Tuned DistilBERT*** </span> 

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100704075-b06d6f80-3359-11eb-92b8-e39b361a8004.png" width="900" height="500">
</p>



<span style="color:blue"> **UN- and SEMI-SUPERVISED LEARNING** </span> 
<br/>
<br/>
<span style="color:orange"> ***FixMatch*** </span> 
text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>


<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100698174-aee97a80-334c-11eb-9839-f85bd999ce0f.png" width="800" height="300">
</p>

text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>
## **Experiments**


<span style="color:blue"> **MULTIMODAL LEARNING: Late Fusion** </span> 
<br/>
We train 3 different late fusion setups. We compare the results with the 
text text text text text text text text text text text text text text text text text text text text text text text text
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100700177-b7907f80-3351-11eb-830f-ab8c4930a67e.png" width="600" height="200">
</p>
<br/>
text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100701939-7ef2a500-3355-11eb-8b44-8c30cd272853.png" width="900" height="500">
</p>
<br/>
text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>
<br/>

<span style="color:blue"> **MULTIMODAL LEARNING: Early Fusion** </span> 
<br/>

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100702240-37204d80-3356-11eb-88b6-d38335ff1bc7.png" width="500" height="300">
</p>
<br/>
text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>

<p align="center">
<img src="https://user-images.githubusercontent.com/7771314/100702866-4eac0600-3357-11eb-80a9-1d53e7989498.png" width="900" height="350">
</p>
<br/>
text text text text text text text text text text text text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text text text text text text
<br/>
<center>
**Modality** | **Olfi et al.** | **Ours**
------------ | --------------- | --------
**Text** | 0.808 | **0.84**
**Image** | 0.833 | **0.89**
**Text + Image** | 0.844 | **0.91**
</center>

## **Label Data Scarcity: Improve the Performance using Pseudo-labels**
<div align="center">
**Model** | **Modality** | **Accuracy**
--------- | ------------ | ------------
**kNN with SupCon** | **Image** | **75.9%**
**DeCLUTR** | **Text** | **73.4%**
**FixMatch** | **Image** | **In progress**
</div>


## **Visualization**

## **The Power of Multimodal Data**



![be39ece1-d2f0-4e87-a95b-fde2bc2bce01](https://user-images.githubusercontent.com/31839963/100531114-36998280-31af-11eb-8fbf-be65227a7168.jpg)


## **Future Work**


- Improve methods/algorithms for obtaining enriched features from text and images
- Improve label propagation techniques
- Automatically generate a coherent summary report about an emergency event

<center>
![unnamed](https://user-images.githubusercontent.com/31839963/100531135-5a5cc880-31af-11eb-9a99-fe46c38032ea.png)
</center>







