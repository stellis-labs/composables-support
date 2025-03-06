import streamlit as st
import requests

API_URL = "http://localhost:5001"  # Flask app port

def show_login_register():
    st.title("Login or Register")

    # Registration form
    st.subheader("Register")
    name = st.text_input("Name", key="register_name")
    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    if st.button('Register'):
        registration_response = requests.post(f'{API_URL}/register', json={
            'name': name,
            'username': username,
            'email': email,
            'password': password
        })
        if registration_response.status_code == 201:
            st.success("Registration successful!")
        else:
            st.error("Registration failed. Please try again.")

    # Login form
    st.subheader("Login")
    login_identifier = st.text_input("Username or Email", key="login_identifier")
    login_password = st.text_input("Password", type="password", key="login_password")
    if st.button('Login'):
        response = requests.post(f'{API_URL}/login', json={
            'identifier': login_identifier,
            'password': login_password
        })
        if response.status_code == 200:
            st.session_state.logged_in = True
            st.session_state.user = login_identifier  # Store user info
            st.rerun()  # Corrected rerun method
        else:
            st.error("Login failed. Please check your credentials.")

def show_welcome_page():
    st.title("Welcome!")
    st.write(f"Hello, {st.session_state.user}! You have successfully logged in.")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()  # Refresh to show login page

def login_page():
    # Initialize the login state if it doesn't exist
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        show_welcome_page()
    else:
        show_login_register()

if __name__ == "__main__":
    login_page()
