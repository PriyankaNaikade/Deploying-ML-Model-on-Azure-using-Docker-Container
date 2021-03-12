from flask import Flask, request, render_template
import pandas as pd
import pickle

from flasgger import Swagger

#This tells from which point we want to start the application, i.e., the control will transfer to main()
app = Flask(__name__)

#Swagger is used to generate the front-end for the application.
Swagger(app)

# Deserialize the pickle object and load the model
pickle_in = open('model.pkl','rb')
model = pickle.load(pickle_in)


# Adding a decorator. This will be the root path for url, i.e., it will trigger this home page first
@app.route('/')
def homepage():
    return render_template("homepage.html")


# Define a function to trigger the model prediction for a single line input or user entered input values
@app.route('/predict_userinput', methods = ['GET'])
def predict_userinput():
	
    """ Predict the Price for the House
	
    ---
    parameters:  
      - name: CRIM
	in: query
	type: number
	required: true
      - name: ZN
	in: query
	type: number
	required: true
      - name: INDUS
	in: query
	type: number
	required: true
      - name: CHAS
	in: query
	type: number
	required: true
      - name: NOX
	in: query
	type: number
	required: true
      - name: RM
	in: query
	type: number
	required: true
      - name: AGE
	in: query
	type: number
	required: true
      - name: DIS
	in: query
	type: number
	required: true
      - name: RAD
	in: query
	type: number
	required: true
      - name: TAX
	in: query
	type: number
	required: true
      - name: PTRATIO
	in: query
	type: number
	required: true
      - name: B
	in: query
	type: number
	required: true
      - name: LSTAT
	in: query
	type: number
	required: true
	
    responses:
        200:
            description: The Output values
	
    """
	
    #Through request get we get the user inputs and then converts the input values from string to float
    CRIM = float(request.args.get('CRIM'))
    ZN = float(request.args.get('ZN'))
    INDUS = float(request.args.get('INDUS'))
    CHAS = float(request.args.get('CHAS'))
    NOX = float(request.args.get('NOX'))
    RM = float(request.args.get('RM'))
    AGE = float(request.args.get('AGE'))
    DIS = float(request.args.get('DIS'))
    RAD = float(request.args.get('RAD'))
    TAX = float(request.args.get('TAX'))
    PTRATIO = float(request.args.get('PTRATIO'))
    B = float(request.args.get('B'))
    LSTAT = float(request.args.get('LSTAT'))
	
    #predict the price for the input values 
    prediction = model.predict([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    
    #return the result
    return "The Price for the House is: " + str(prediction)

	
# Define a function to trigger model predictions for a file with multiple records/rows of input features	
@app.route('/predict_fileinput', methods = ['POST'])
def predict_fileinput():
	
    """ Predict the Price for the House.
	
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true 
	
    responses:
        200:
            description: The output values	
	
    """
	
    #Read the input file and convert it to a dataframe
    df_test = pd.read_csv(request.files.get("file"))
    predictions = model.predict(df_test)
    
    return "The Price for the " + str(len(predictions)) + " Houses are: " + str(list(predictions))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
