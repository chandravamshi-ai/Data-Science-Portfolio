import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('Radom forest classifier ')

st.info('predicts species of penguines')

with st.expander('Data'):
  st.write('**Penguinue Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species',axis=1)
  X_raw
  
  st.write('**y**')
  y_raw = df.species
  y_raw

  

with st.expander('Scatter plot for bill length and body mass. Colored using species type'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

# input features
with st.sidebar:
  # "species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  st.header('Input features')
  island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  gender = st.selectbox('Gender',('male','female'))
  bill_length_mm = st.slider('Bill length (mm)',32.1,59.6,43.9)
  bill_depth_mm = st.slider('Bill depth (mm)',13.1,21.5,17.2)
  flipper_length_mm = st.slider('Flipper lenght (mm)',172.0,231.0,201.0)
  body_mass_g = st.slider('Body mass (g)',2700.0,6300.0,4207.0)


  # create df for input features
  data = {
    'island':island,
    'bill_length_mm':bill_length_mm,
    'bill_depth_mm':bill_depth_mm,
    'flipper_length_mm':flipper_length_mm,
    'body_mass_g':body_mass_g,
    'sex':gender 
  }
  
  

# data preparation
input_df= pd.DataFrame(data,index=[0])
input_penguins = pd.concat([input_df,X_raw],axis=0)

# encode categroical features
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins,prefix=encode)
X = df_penguins[1:]
input_row= df_penguins[:1]

# encode y
target_mapper = {'Adelie':0,'Chinstrap':1,'Gentoo':2}

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Input features'):
  
  st.write('inputs from sidebar')
  input_df
  st.write('combined data(from siderbar values and actual dataset values)')
  input_penguins
  
with st.expander('Data Preparation'):
  st.write('encoded x')
  input_row
  st.write('encoded y')
  y

# model training and inference
# training
rfclf = RandomForestClassifier() 
rfclf.fit(X,y)

# apply model to make predictions
prediction = rfclf.predict(input_row)
prediction_proba = rfclf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)

df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                               1: 'Chinstrap',
                              2: 'Gentoo'})
df_prediction_proba

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = ['Adelie', 'Chinstrap', 'Gentoo']
st.success(str(penguins_species[prediction[0]]))
