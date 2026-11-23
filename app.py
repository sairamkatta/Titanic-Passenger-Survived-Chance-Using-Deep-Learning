import streamlit as st
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model
import pickle

st.title('Passenger Survival Chance in the Titanic Journey')

Pclass=st.slider("Enter the Passenger class for the user", 1,3)
sex=st.selectbox("Enter the Passenger Gender",['male','female'])
sibsp=st.slider("Enter the Passenger total number of Sibling and Spouse",1,8)
parch=st.slider("Enter the Passenger total number of Parents and Child",0,6)
fare=st.number_input("Enter the Fare of the Passenger")
embarked=st.selectbox("Enter the Passenger station from where they started the jouney",['Southampton','Chebourg','Queenstown'])

data=pd.DataFrame([{'Pclass':Pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])


modal=load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)
    
with open('scaler_encoder.pkl','rb') as file:
    scaler=pickle.load(file)
    
data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=modal.predict(data)

y=y[0][0]

def Chance(y):
    if y>0.5:
        return 'The Passenger will Survive the Journey'
    else: 
        return 'The Pasenger wont survive the Journey'

if st.button('Predict Survival Chance'):
    st.write('Probability of Passenger Survival Chance', y)
    st.write(Chance(y))