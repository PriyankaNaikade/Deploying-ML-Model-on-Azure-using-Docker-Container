#Specifying parent base image as Python3.8.5 
FROM python:3.8.5-slim


#Define the directory '/dhpriceapp' inside the docker container as the working directory of app
WORKDIR /dhpriceapp

# Copying all the files (model files, requirements file) from current project directory to a new directory '/pricepredapp' in docker container
COPY . /dhpriceapp


#Installing the dependencies of the flask web application
RUN pip3 install --no-cache-dir -r requirements.txt

#Exposing default port for the web application
EXPOSE 5000

#CMD command launches the application when the container starts running.
CMD ["python", "app.py"]