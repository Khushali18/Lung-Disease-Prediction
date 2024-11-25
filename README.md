# README.txt

## Aim
The goal of this project is to classify X-ray images between 3 classes namely Lung Opacity, Normal and Viral Pneumonia.

To develop this classification pipeline, we require an Lung X-ray dataset containing 3 Classes mentioned in Aim. This raw data is downloaded and is available in the **Data** folder.

As we know, the quality of data is crucial for any machine learning model. By providing high-quality data that is understandable by the model or algorithm, we can achieve optimal performance. Therefore, we must analyze and preprocess this data. The implementation of these processes can be found in the **Preprocessing** folder.

With the preprocessed data ready, we can input it into the models most suitable for our dataset. Based on our analysis during the preprocessing task, we have selected the CNN and ResNet models. The implementation of these models can be found in the **Models** folder.

The **Raw_Pipeline.ipynb** file contains the complete implementation, covering everything from data exploration and preprocessing to model training and evaluation.

The **Plan.docx** outlines the action plan for the project.

## Folder Structure

Unstructure_Data_Pipeline
 |
 |---Data (Extracted from Kaggle: https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease)
 |
 |---Models
 |     |
 |     |---CNN
 |     |
 |     |---ResNet
 |
 |---Preprocessing
