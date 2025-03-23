import streamlit as st
from catboost import CatBoostRegressor
from math import log
import numpy as np

@st.cache_resource
def load_model(model_path):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

def main():
    st.title("House Price Prediction App")
    st.write("This app predicts the house price based on the given parameters.")

    model = load_model("model.cbm")

    st.sidebar.header("Input Parameters")
    area = st.sidebar.slider("Area (m^2)", min_value=100, max_value=1500, value=480)
    bed = st.sidebar.number_input("#Bedrooms", min_value=1, max_value=6, value=2)
    bath = st.sidebar.number_input("#Bathrooms", min_value=1, max_value=4, value=1)
    fl = st.sidebar.number_input("#Floors", min_value=1, max_value=4, value=2)
    pk = st.sidebar.number_input("#Parkings", min_value=0, max_value=3, value=0)
    furn_vals = ["furnished", "semi-furnished", "unfurnished"]
    furn = st.sidebar.selectbox("Furnishing", furn_vals)
    furn_ohe = [fv == furn for fv in furn_vals]
    assert sum(furn_ohe) == 1

    arft = area / 0.09290304
    log_arft = log(arft)
    num_feat = np.array([arft, log_arft, bath, bed, fl, pk])
    cat_values = ['yes', 'no', 'no', 'no', 'no', 'no']
    cat_feat = np.concatenate([[0, 1] if vl == 'yes' else [1, 0] for vl in cat_values] + [furn_ohe])

    feat = np.concatenate([num_feat, cat_feat])[None, :]
    if st.sidebar.button("Predict Price"):
        prediction = model.predict(feat)
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")


if __name__ == "__main__":
    main()
