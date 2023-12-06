import pickle 
import os 
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np 
import pandas as pd 
app =Flask(__name__)  #starting point of my app 

model = pickle.load(open('model.pkl','rb'))

# / means home page 
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])

#capture what is present inside 'data' (the input we are going to give ) using request in json format 
#and store in data variable 
#so as soon as u hit the predict.api ur input will be loaded using request in json format 
def predict_api():
    data=[float(x) for x in request.form.values()]
    print(data)
    #print(np.array(list(data.values())).reshape(1,-1)) #1,-1 to say ur values are single input 
    t = np.array(data).reshape(1,-1)

    output = model.predict(t)
    if output==0:
        txt = 'hmm. he is not having heart disease '
    else:
           txt = 'oh. he is having heart disease '

    return render_template("home.html",prediction_text="{}".format(txt))
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)


