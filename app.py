from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf



BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    """Load the existing trained model and its matching preprocessing objects."""
    model = tf.keras.models.load_model(BASE_DIR / "model.h5", compile=False)
    with (BASE_DIR / "onehot_encoder_geo.pkl").open("rb") as file:
        onehot_encoder_geo = pickle.load(file)
    with (BASE_DIR / "label_encoder_gender.pkl").open("rb") as file:
        label_encoder_gender = pickle.load(file)
    with (BASE_DIR / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    return model, onehot_encoder_geo, label_encoder_gender, scaler


model, onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()

   
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])




# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## One hot encoding 
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography']),
)




## Combine one-hot encoding columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)



## Scale input data
feature_names = getattr(scaler, "feature_names_in_", None)
if feature_names is not None:
    # Keep inference columns in the exact order used when fitting the scaler.
    input_data = input_data.reindex(columns=feature_names)
input_scaled = scaler.transform(input_data)



## Predict churn

prediction = model.predict(input_scaled, verbose=0)
prediction_prob = float(prediction[0][0])


st.write(f'Churn Probability:{prediction_prob:.2f}')



# Why does it return [[0.85]] instead of just 0.85?

# Because TensorFlow returns predictions in a 2D NumPy array:
# - First dimension = Number of input samples (customers).
# - Second dimension = Number of output values per sample.


if prediction_prob>0.5:
    st.write("This customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")    
