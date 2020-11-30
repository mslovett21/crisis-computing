<p align="center"> # CRISIS COMPUTING: Multimodal Social Media Content for Improved Emergency Response </p>

# Motivation

**Multimodal** data shared on Social Media during critical emergencies often contains useful information about a scale of the events, victims and infrastructure damage.

![1f976786-3b50-429d-bf0c-d11ab5e85c6f](https://user-images.githubusercontent.com/31839963/100529409-4360ab00-319c-11eb-804e-223d0418ffb6.jpg)



# PROBLEM STATEMENT

- Explores different techniques of representation learning to improve performance on nuance classification of informative vs non-informative social media post in domain of crisis computing.
- Investigate applications of unsupervised and semi-supervised learning methods to mitigate the issue of labeled data scarcity on the classification task.

![a9cdfe42-149f-4467-985a-3cc870b8c6c4](https://user-images.githubusercontent.com/31839963/100529467-f16c5500-319c-11eb-93da-a4dbf7c5a17e.jpg)



# METHODS


## CONTRASTIVE LEARNING


### SupCon Architecture

Goal is to learn good representations of the classes first so that later it becomes easy to classify them.

![5e85a5d5-9031-447c-b4ef-49bc7661b89a](https://user-images.githubusercontent.com/31839963/100529533-735c7e00-319d-11eb-8c41-1a2eab583afd.jpg)



## SEMI-SUPERVISED LEARNING

### Architectures for text

**DeCLUTR** | **Fine-Tuned DistiBERT**
----------- | ------------------------
![decltr](https://user-images.githubusercontent.com/31839963/100529619-3b096f80-319e-11eb-8624-11dfa551f3d8.jpg) | ![FTD](https://user-images.githubusercontent.com/31839963/100529623-48265e80-319e-11eb-94ff-8349ec130169.jpg)

Goal is to learn high-quality universal sentence embeddings to label the unlabeled tweets.


### FixMatch

Goal is to provide a pseudo-label to the unlabeled images to retain/fine-tune the existing model by using augmented images for consistency regularization and pseudo-labelling them.

![5c67944c-8d8f-488d-a67f-7bbc69d4e588](https://user-images.githubusercontent.com/31839963/100529666-af441300-319e-11eb-9035-4d3e0f0b70f9.jpg)



# EXPERIMENTS


## LATE FUSION


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


## EARLY FUSION


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



# FUTURE SCOPE


- Improve methods/algorithms for obtaining enriched features from text and images
- Improve label propagation techniques
- Automatically generate a coherent summary report about an emergency event


![unnamed](https://user-images.githubusercontent.com/31839963/100531135-5a5cc880-31af-11eb-9a99-fe46c38032ea.png)








