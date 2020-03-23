# 2018 Data Science Bowl 

Prediction for 2018 Data Science Bowl challange.

# Overview

Semantic segmentation model with UNet architecture using Keras. To get data and more ifnormation can be found at [Kaggle 2018 Data Science Bowl challange][df1]

### Data Managment

Once downloaded unzip relavent archives so that structure looks like this:
data-science-bowl-2018
|__Basic_EDA.ipynb
|__predict_masks.py
|__stage1_sample_submission.csv
|__stage1_solution.csv
|__stage1_test
__|__0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5
_____|__images
________|__0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5.png
    ...
|__stage1_train
__|__0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9
____|__images
______|__0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png
____|__masks
______|__0adbf56cd182f784ca681396edc8b847b888b34762d48168c7812c79d145aa07.png
        ...
### Preprocessing

For preprocessing images are resized to the same squares (128x128 pixels by default). For further data augmentation one might apply other preprocessing and transformations (blurring, mirroring, etc).

### Model

The model replecates the basic structure of [U-Net model][unet] with reduced number neurons which still yields adequate predictions.

   [df1]: <https://www.kaggle.com/c/data-science-bowl-2018/data>
   [unet]: <https://arxiv.org/pdf/1505.04597.pdf>
   
### Notes

Scripts where developed and ran using Google Colab Notebook (Basic_EDA with Jupyter Notebook). It was not tested localy (due to technical difficalties with tensorflow backend). Replaced behavior can be obtained by running train script from Google Colab.
