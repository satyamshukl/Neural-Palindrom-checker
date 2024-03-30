import numpy as np

SEED=18
np.random.seed(SEED)

import pandas as pd
from model import Palindrome
import streamlit as st

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center;'>Palindrome Prediction</h1> <br><h3>Enter the binary string below:</h3>", unsafe_allow_html=True)

if __name__ == '__main__':
    model = Palindrome()
    
    user_input = st.text_input("")
    caption_button = st.button("Predict")
    if caption_button:
        st.balloons()
        if user_input:
            input = np.array([int(i) for i in user_input])
            assert len(input) == 10

            for x in input:
                if x not in [0, 1]:
                    raise AssertionError
            model.load_weights('palindrome_weights.pkl')
            predict = model.predict(input)
            if predict == 1:
                st.markdown("<h2 style='text-align: center;'>The given input is Palindrome</h2>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align: center;'>The given input is NOT a Palindrome</h2>",
                            unsafe_allow_html=True)
        else:
            print("Enter a binary string")