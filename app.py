import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)  # This will be starting point of my application where it will run
regmodel=pickle.load(open('regmodel.pkl','rb')) # Load the model
scalar=pickle.load(open('scaling.pkl','rb')) # Load the standardizing pickle file

@app.route('/')  # this is the first root. localhost/ means it should definitely go to homepage
def home():
    return render_template('home.html')

# Lets create a predict api. This is a POST request where data will be collected using POSTMAN.
# The data will be passed through scalar transformation and then through a regression model which gives the o/p
@app.route('/predict_api',methods=['POST']) 
def predict_api():
    data=request.json[data]  # whatever input we give for prediction will be in json format and saved in data variable.
    print (data)
    # Once we receive the data,first thing we did was standardizing the data.
    # We have pickle file for model but not for standardization
    # So we have created a pickle file (scaling.pkl) for standardizing
    # Json data will be in key value pair format. We will take values and convert to list. Then we need to reshape (Refer ipynb notebook pickle part)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])  # usually we get o/p as a 2d array. we will take first value. Refer ipynb note
    return jsonify(output[0])
    

# Instead of creating an api, we will create a small webapp itself which receives some inputs and after submission it gives o/p
# USer will manually inputs all the individual feature values through an html form. From there we will get the data here at app.py

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)