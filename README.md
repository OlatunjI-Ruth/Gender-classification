# Project: Gender classification

# Project Goal
The goal of this project is to train a model that accurately predicts the gender of a person in an image, from data collection stage to deployment.

# Dataset Information.
This dataset was collected by scrapping images of the male and female gender from google images. the code for the task was written with the assistance of "Tech with Tom" on youtube https://www.youtube.com/watch?v=NBuED2PivbY

# Tools and Dependencies
Python
Selenium
Chrome web driver
requests
Google Colab
Tensorflow
Keras
Matplotlib
Numpy
Flask
Heroku
CNN

# Steps Involved
1. Using the scrape.py code, google images was crawled and a total of 4,500 images of both the male and female gender with the use of selenium, chrome web driver and requests. The images were sorted, invalid image formats were deleted and the valid images were saved to their respective directories.

2. The folder was uploaded to my google drive, which was mounted to my colab.

3. The important dependencies were imported and the images were preprocessed using ImageDataGenerator.

4. Due to the not so large amount of images I was able to scrape, I applied the Data argumentation and Transfer Learning Techniques to train my model, of which I got a 15% loss and 90% training and validation accuracy.

5. The model was deployed using Heroku, which predicts 8 of 10 images correctly.

for a well detailed explanation of the steps and errors involved, please visit my blog https://medium.com/@rutherfordola

Here is a link to the web app www.gender-classificationn.herokuapp.com
