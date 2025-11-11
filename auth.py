import streamlit as st
import jwt
import datetime
from database import verify_user, create_user

# Secret key for JWT - in production use proper secret management
SECRET_KEY = "irmc-aura-secret-key-2024"

def login_user(email, password):
    user_id = verify_user(email, password)
    if user_id:
        # Create session token
        payload = {
            'user_id': user_id,
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=5)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        
        # Store in session state
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.email = email
        st.session_state.token = token
        return True
    return False

def signup_user(email, password):
    return create_user(email, password)

def logout_user():
    for key in ['logged_in', 'user_id', 'email', 'token']:
        if key in st.session_state:
            del st.session_state[key]

def check_session():
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        try:
            # Verify token is still valid
            jwt.decode(st.session_state.token, SECRET_KEY, algorithms=['HS256'])
            return True
        except jwt.ExpiredSignatureError:
            logout_user()
    return False
