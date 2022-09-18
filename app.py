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

# Lets create a predict api
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
    print(output[0])
    return jsonify(output[0])
    




if __name__=="__main__":
    app.run(debug=True)