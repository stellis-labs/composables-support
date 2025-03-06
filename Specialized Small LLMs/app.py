import streamlit as st
from frontend.login import login_page
from frontend.dashboard import dashboard_page

def main():
    # ... existing code ...
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        dashboard_page()
    else:
        login_page()

if __name__ == "__main__":
    main() 