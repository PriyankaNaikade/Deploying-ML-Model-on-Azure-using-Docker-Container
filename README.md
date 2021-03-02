# Deploying-ML-Model-on-Azure-using-Docker-Container
Involves Building an ML model, Creating an API for the model using Flask, Dockerizing Flask Web Application, and Deploying on Azure Cloud Platform.




## Overview
This repo demonstrates how to deploy a machine learning model on Azure as a webservice using Flask and Docker Container.




## Description of Files
1. House Price Prediction Folder
      - **pricepredmodel.py** - is used for building the machine learning model for a house price prediction problem. We are using a multiple linear regression model for making the predictions. The pickle version of the model (model.pkl) is generated by this file. 
      - **app.py** - In this file, we are developing a web application using Swagger for front-end GUI and Flask as a backend to create an API or serve the ML model as a REST API.
      - **requirements.txt** - In this file, we specify the dependencies of the application, i.e., list of modules and its specific version.
      - **Dockerfile** - This file consists of set of instructions needed to create a docker image, when run generates its live instance alias docker container (that will package the applications and its dependencies together).

2. Test Data File Folder consists of a test file (with 2 sample records) for testing the application.

3. Documentation Folder consists of a file describing the end-to-end ML deployment procedure in a detailed manner.




## Set Up and Test Application Locally
1. Clone the 'House-Price-Prediction' folder in your local system using git command: *``` git clone ```*.

2. You can run the pricepredmodel.py file using **```python pricepredmodel.py```** command which will generate the model.pkl file in your current working directory or you can use the trained model - 'model.pkl' present in folder.

3. To test the Web Application, run **```app.py```** command. Navigate to the URL (should be http://0.0.0.0:5000) displayed in the output console. It should open up a web page displaying "Welcome to the Home Page". (Note: Make sure that the model.pkl and app.py files are located in the same directory).

4. We can test the API and make predictions using curl requests. Or we can use POSTMAN (https://www.getpostman.com/) to make prediction for a single user input data by sending a GET request to http://localhost:5000/predict_userinput/ and make predictions for a file input by sending a POST request to http://localhost:5000/predict_fileinput url.
(For  more details please refer the file present inside the documentation folder). 

5. We can also test the application and make price predictions on the web browser by navigating to http://localhost:5000/apidocs/. (For  more details please refer the file present inside the documentation folder).




## Containerize the Flask Web App using Docker and Test locally
Basic purpose of Containerization: To package the code with all its libraries and dependencies into a single container so that the application works the same way when it is moved from one computing environment to the other without , say from dev -> test/prod environment. 

#### Dockerizing the Flask Web App
1. Install the docker desktop app and make sure it works fine locally.

2. Navigate to your working directory (all the files should exist in the same folder - Dockerfile, requirements, model app files)

3. Build the Docker by running the build command: 
            
           docker build -t image-name .
   
   Docker image will be created. Check using **```docker images```** command.

4. Run the Docker Image using: 
            
            docker run -p 5000:5000 image-name
            
   To run in deattach mode include **```-d```**.

5. When the docker engine runs this docker image, a container will be created which will be in running status. Check using **```docker ps```** command.

6. Now, we can test the app by navigating to http://localhost:5000/apidocs/

7. Once you stop the container, the web app will no longer open. To stop the docker image use: 
    
            docker stop image-name





## Host the Dockerized Flask Web App on Azure

#### Publish the Docker Container to Docker Hub
We have to push the docker container that we created in our local system to a public repository in docker hub, so that it can be accessed from anywhere and finally deploy it on the cloud. Else we would have to create a new container again in Azure.
To do this, Refer Step-6 in the file present in the Documentation folder.

#### Create a Web App Service on Azure Portal
1. Login to Azure Portal. On the home page, select 'Create a resource' and then search & select Web App service for hosting the docker container that we pushed in the hub. 

2. Enter 'Project Details' by seleting an appropriate plan and resource group.

3. Enter 'Instance Details' -
      - Publish - Docker
      - OS - Linux

4. Navigate to Docker and provide the its details
        - Image Source - Docker Hub
        - Image and Tag - <Depending on whatever you mentioned in the previous step while pushing the container in docker hub>

5. Under Tags, Add PORT field and specify the port number as 5000, since that is what we have used for the application.

6. Click 'Review+Create'  -> 'Create'

7. Once the deployment is done -> 'Go to Resource' -> Check the URL and test it. It should open 'Welcome to Home Page'

8. To use the app for making the house price predictions, append '/apidocs/' at the end of the url, i.e., "[https://***********.azurewebsites.net/apidocs/](url)".

9. Once you're done with testing the application/making predictions, make sure to stop the service (You do not want to use up all the resources).

For more details regarding this part, please refer Step-7 in the file present in the Documentation folder




