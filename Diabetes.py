import numpy as np
import pickle

loaded_model=pickle.load(open('C:/code_playground/ML/STREAMLIT/trained_model.sav','rb'))

input_dat=(4,110,92,0,0,37.6,0.191,30)

inp_as_np= np.asanyarray(input_dat)
input_dat_reshape=inp_as_np.reshape(1,-1)

eval_predict=loaded_model.predict(input_dat_reshape)

#print(eval_predict)
if (eval_predict[0]==0):
  print('not diabetic')
else:
  print('diabetic')