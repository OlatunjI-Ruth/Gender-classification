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
Using the scrape.py code, google images was crawled and a total of 4,500 images of both the male and female gender with the use of selenium, chrome web driver and requests. The images were sorted, invalid image formats were deleted and the valid images were saved to their respective directories.

2. The folder was uploaded to my google drive, which was mounted to my colab.

3. The important dependencies were imported and the images were preprocessed using ImageDataGenerator.

4. Due to the not so large amount of images I was able to scrape, I applied the Data argumentation and Transfer Learning Techniques to train my model, of which I got a 15% loss and 90% training and validation accuracy.

5. The model was saved, and pushed to GitHub using the git bash due to the large weight of the model’s weight.

6. My GitHub was connected to Heroku, and my model was deployed which could predict 8 out of 10 images correctly.

The downside is, due to the fact that my files are quite large, after predicting 4 images, I’d get a Heroku “Memory quota vastly exceeded” error.

A full documentation on how I went about the project from web scraping to deployment, the tools and libraries used, errors encountered, how they were resolved, all links to resources I used and my deployed web, will be out this week hopefully. I am just putting this post out here because I really am not loud about my little wins, so this will motivate me to actually write out the full details this week.

Below is a screen record of me, testing the app. If you have the luxury of time and patience, please watch till the end
