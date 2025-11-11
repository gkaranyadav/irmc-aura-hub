import streamlit as st
import jwt
import datetime
from database import verify_user, create_user

# Secret key for JWT
SECRET_KEY = "irmc-aura-secret-key-2024"

def generate_token(user_id, email):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=5),
        'iat': datetime.datetime.utcnow()
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def verify_token(token):
    """Verify JWT token and return user data if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def login_user(email, password):
    """Login user and set session state"""
    user_id = verify_user(email, password)
    if user_id:
        token = generate_token(user_id, email)
        
        # Store in session state
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.email = email
        st.session_state.token = token
        
        # Also store in persistent session state
        st.session_state.persistent_login = True
        st.session_state.persistent_email = email
        st.session_state.persistent_token = token
        
        return True
    return False

def signup_user(email, password):
    """Create new user account"""
    return create_user(email, password)

def logout_user():
    """Logout user and clear all storage"""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def check_session():
    """Check if user is logged in"""
    # Method 1: Check if already logged in this session
    if st.session_state.get('logged_in'):
        try:
            jwt.decode(st.session_state.token, SECRET_KEY, algorithms=['HS256'])
            return True
        except:
            pass
    
    # Method 2: Check persistent session (survives refreshes)
    if st.session_state.get('persistent_login'):
        token = st.session_state.get('persistent_token')
        email = st.session_state.get('persistent_email')
        
        if token and email:
            payload = verify_token(token)
            if payload:
                # Restore session
                st.session_state.logged_in = True
                st.session_state.user_id = payload['user_id']
                st.session_state.email = payload['email']
                st.session_state.token = token
                return True
    
    return False
