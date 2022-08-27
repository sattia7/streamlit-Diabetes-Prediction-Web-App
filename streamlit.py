import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('C:/code_playground/ML/STREAMLIT/trained_model.sav','rb'))

def diabetes (input_data):
    

    inp_as_np= np.asanyarray(input_data)
    input_dat_reshape=inp_as_np.reshape(1,-1)

    eval_predict=loaded_model.predict(input_dat_reshape)

    print(eval_predict)
    if (eval_predict[0]==0):
        return 'not diabetic'
    else:
        return 'diabetic'

def main():

    st.title('Diabetes Prediction Web Page')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure')
    SkinThickness = st.text_input('SkinT hickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age=st.text_input('Age')

    Diagnosis=''

    if st.button('Diabetes Test Result'):
        Diagnosis= diabetes([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(Diagnosis)

if __name__ == '__main__':
    main()    


    


