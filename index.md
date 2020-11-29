# MOTIVATION

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

Goal is to learn high-quality universal sentence embeddings to label the unlabeled tweets.

### DeCLUTR

![decltr](https://user-images.githubusercontent.com/31839963/100529619-3b096f80-319e-11eb-8624-11dfa551f3d8.jpg)

### Fine-Tuned DistiBERT

![FTD](https://user-images.githubusercontent.com/31839963/100529623-48265e80-319e-11eb-94ff-8349ec130169.jpg)

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

**Modality** | **Accuracy** 
------------ | ------------
 ****| **Olfi et al.** | **Ours**
------------ | --------------- | --------
**Text** | 0.808 | **0.84**
**Image** | 0.833 | **0.89**
**Text + Image** | 0.844 | **0.91**





## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/mslovett21/crisis-computing/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mslovett21/crisis-computing/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
