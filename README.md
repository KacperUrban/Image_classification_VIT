# Image_classification_VIT
# General info
In this repository I will use different pretrained Vision Transfomers to classifies different buildings and places in Rzeszów. 
# Dataset
The Beautiful Rzeszow dataset consists of 3000 images of 50 tourist sites in Rzeszow, Poland. Each attraction, such as a building, monument, or place, was photographed at a different time of the day (day and night) and season (spring, autumn, and winter). In the dataset, there are large scale, viewpoint and rotation variations, as well as challenging illumination conditions. All with a variety of occlusions.
# Models
In this project I will use a three open access models:
* Google ViT - https://huggingface.co/google/vit-base-patch16-224,
* Microsoft ViT - https://huggingface.co/microsoft/beit-base-patch16-224,
* Apple ViT - https://huggingface.co/apple/mobilevit-small.

# Experiments
This table describe all test which was conducted. The first column depicte what fraction of data was in test set. For example in the first row the test set include all images taken in a day. As we can see the best result I got with Google Vit and Microsoft Vit. The result from Apple Vit was worsed, but we have to remember that is the smallest model. All as we can see the test set substianaly impact models performance. The best performance occur on random test set, because models got diverse examples in train and test set.

|              | Google ViT Accuracy | Google ViT F1 | Microsoft ViT Accuracy | Microsoft ViT F1 | Apple ViT Accuracy | Apple ViT F1 |
|--------------|--------------------|---------------|---------------------|---------------|----------------|------------|
| Day         | 0.036667           | 0.034275      | 0.037333            | 0.03457       | 0.026          | 0.026301   |
| Night       | 0.039333           | 0.039146      | 0.0333              | 0.036012      | 0.045333       | 0.032903   |
| Winter      | 0.041              | 0.041         | 0.04                | 0.04          | 0.033          | 0.030102   |
| Spring      | 0.04               | 0.03856       | 0.04                | 0.039512      | 0.037          | 0.036283   |
| Autumn      | 0.037              | 0.038435      | 0.037               | 0.038841      | 0.04           | 0.042282   |
| Day-Winter  | 0.04               | 0.04          | 0.04                | 0.039048      | 0.034          | 0.030678   |
| Night-Winter| 0.042              | 0.041905      | 0.04                | 0.04          | 0.04           | 0.036088   |
| Day-Spring  | 0.04               | 0.039048      | 0.04                | 0.039048      | 0.036          | 0.034138   |
| Night-Spring| 0.042              | 0.040529      | 0.04                | 0.04          | 0.036          | 0.031902   |
| Day-Autumn  | 0.04               | 0.04          | 0.038               | 0.038         | 0.036          | 0.037118   |
| Night-Autumn| 0.034              | 0.037157      | 0.036               | 0.036825      | 0.04           | 0.039468   |
| Random      | 0.98               | 0.97962       | 0.98                | 0.97857       | 0.78667        | 0.78119    |

# Future improvements
I plan to create a small application with FastAPI and probably with streamlit to enable easy usage of trained models.

# Status
The project is on going.
