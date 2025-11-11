import streamlit as st
import jwt
import datetime
import extra_streamlit_components as stx
from database import verify_user, create_user

# Secret key for JWT
SECRET_KEY = "irmc-aura-secret-key-2024"

def get_cookie_manager():
    """Get cookie manager"""
    return stx.CookieManager()

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
    """Login user and set cookie"""
    user_id = verify_user(email, password)
    if user_id:
        token = generate_token(user_id, email)
        cookie_manager = get_cookie_manager()
        
        # Set cookie that expires in 5 hours
        cookie_manager.set("auth_token", token, max_age=18000)  # 5 hours in seconds
        
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
    """Logout user and clear cookie"""
    cookie_manager = get_cookie_manager()
    cookie_manager.delete("auth_token")
    
    # Clear session state
    keys_to_clear = ['logged_in', 'user_id', 'email', 'token']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def check_session():
    """Check if user is logged in via session or cookie"""
    # Check session state first
    if st.session_state.get('logged_in'):
        try:
            jwt.decode(st.session_state.token, SECRET_KEY, algorithms=['HS256'])
            return True
        except:
            pass
    
    # Check cookie for auto-login
    return check_cookie_auth()

def check_cookie_auth():
    """Check if valid auth token exists in cookie"""
    cookie_manager = get_cookie_manager()
    token = cookie_manager.get("auth_token")
    
    if token:
        payload = verify_token(token)
        if payload:
            # Auto-login from cookie
            st.session_state.logged_in = True
            st.session_state.user_id = payload['user_id']
            st.session_state.email = payload['email']
            st.session_state.token = token
            return True
        else:
            # Invalid token - clear cookie
            cookie_manager.delete("auth_token")
    
    return False
