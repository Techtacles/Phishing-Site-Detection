from flask import Flask,request, jsonify, render_template
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import pickle
scaler=pickle.load(open("scale.pkl","rb"))
app = Flask(__name__)
model=tf.keras.models.load_model("model1.h5")

def norm(x):
    return scaler.transform(x)
def checkSpecial(url):
    """Returns number of special characters in string"""
    regex = re.compile('[@_!#$%^&*()<>?|}{~]')
    return len([c for c in url if regex.search(c)])

def getNums(url):
    """Returns number of digits in string"""
    return len([c for c in url if c.isdigit()])

def entropy(url):
    """Returns entropy of string"""
    s = url.strip()
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    ent = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
    return ent

def numSubDomains(url):
    """Returns number of subdomains in the given URL"""
    subdomains = url.split('http')[-1].split('//')[-1].split('/')
    return len(subdomains)-1



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])

def predict():
   # int_features=[float(x) for x in request.form.values()]
    url=[request.form["url"]]
   # a=["http://www.seroogys.com/..."]
    len_num=np.sum([len(p) for p in url[0]])
    num=getNums(url[0])
    special=checkSpecial(url[0])
    has_per=np.sum([1 if ("%" in q) else 0 for q in url[0]])
    ent=entropy(url[0])
    subdomain=numSubDomains(url[0])
    final_list=np.array([len_num,num,special,has_per,ent,subdomain])
    final_list1=norm([final_list]).tolist()
    pred=model.predict([final_list1])
    pred2=np.round(pred)
    pred3=int(pred2[0][0][0])
    if pred3==1:   
        return render_template("index.html",prediction_text=f"A phishing site")
    else:
        return render_template("index.html",prediction_text=f"A legitimate site")
        
    
  


if __name__ == '__main__':
   app.run(debug=False)