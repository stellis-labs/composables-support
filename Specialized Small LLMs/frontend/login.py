import streamlit as st
import requests

def login_page():
    # ... existing code ...
    if st.button('Login'):
        response = requests.post('http://localhost:5000/login', json={'username': username, 'password': password})
        # ... existing code ...

# ... existing code ... 