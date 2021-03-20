#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
from flask import Flask,render_template,request,redirect
import pickle 


# In[71]:


app=Flask(__name__)

model=pickle.load(open('random_forest.pkl','rb'))
scaler=pickle.load(open('AQI_scaler_.pkl','rb'))

@app.route("/")
def base():
    return render_template('AQI_Base.html')

@app.route("/AQI_home") 
def home():
    return redirect('/')

@app.route("/AQI_form")
def signup_form():
    return render_template('AQI_form.html')

@app.route("/AQI_Result",methods=['POST'])
def pred():
    
    
    '''
    avg_temp=request.form['fname']
    max_temp=request.form['lname']
    min_temp=request.form['min_temp']
    slp=request.form['slp']
    Humidity=request.form['Humi']
    avg_visibility=request.form['vv']
    avg_wind_speed=request.form['v']
    max_wind_speed=request.form['vm']
    '''
    
    features=[float(x) for x in request.form.values()]
    
    final_features=[np.array(features)]
    
    final_features=scaler.transform(final_features)
    
    print(final_features)
    
    prediction=model.predict(final_features)
    
    print(prediction)
    
    print(features)
    
    return render_template('AQI_Result.html',outcome="%0.2f" %prediction[0])
    
    


if __name__=='__main__':
    app.run()


# In[48]:





# In[ ]:




