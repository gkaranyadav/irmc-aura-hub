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
    """Login user"""
    user_id = verify_user(email, password)
    if user_id:
        token = generate_token(user_id, email)
        
        # Store in session state
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.email = email
        st.session_state.token = token
        
        return True
    return False

def signup_user(email, password):
    """Create new user account"""
    return create_user(email, password)

def logout_user():
    """Logout user"""
    # Clear all session state
    keys_to_clear = ['logged_in', 'user_id', 'email', 'token']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def check_session():
    """Check if user is logged in"""
    # Simple check - just see if logged_in exists and is True
    return st.session_state.get('logged_in', False)
