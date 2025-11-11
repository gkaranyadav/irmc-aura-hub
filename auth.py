import streamlit as st
import jwt
import datetime
import json
from database import verify_user, create_user

# Secret key for JWT - in production use proper secret management
SECRET_KEY = "irmc-aura-secret-key-2024"  # We'll move this to secrets later

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
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

def login_user(email, password):
    """Login user and set session + local storage"""
    user_id = verify_user(email, password)
    if user_id:
        # Generate token
        token = generate_token(user_id, email)
        
        # Store in session state
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.email = email
        st.session_state.token = token
        
        # Store in local storage via JavaScript
        store_token_in_local_storage(token, email)
        return True
    return False

def signup_user(email, password):
    """Create new user account"""
    return create_user(email, password)

def logout_user():
    """Logout user and clear all storage"""
    # Clear session state
    for key in ['logged_in', 'user_id', 'email', 'token']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear local storage via JavaScript
    clear_local_storage()

def check_session():
    """Check if user is logged in (session + local storage)"""
    # First check session state
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        try:
            # Verify token is still valid
            jwt.decode(st.session_state.token, SECRET_KEY, algorithms=['HS256'])
            return True
        except jwt.ExpiredSignatureError:
            logout_user()
    
    # If no session, check local storage
    return check_local_storage_token()

def store_token_in_local_storage(token, email):
    """Store token in browser local storage using JavaScript"""
    js_code = f"""
    <script>
        localStorage.setItem('auth_token', '{token}');
        localStorage.setItem('user_email', '{email}');
        localStorage.setItem('login_time', '{datetime.datetime.now().isoformat()}');
    </script>
    """
    st.components.v1.html(js_code, height=0)

def clear_local_storage():
    """Clear tokens from browser local storage"""
    js_code = """
    <script>
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_email');
        localStorage.removeItem('login_time');
    </script>
    """
    st.components.v1.html(js_code, height=0)

def check_local_storage_token():
    """Check if valid token exists in local storage"""
    # Create JavaScript to get token from local storage
    js_code = """
    <script>
        const token = localStorage.getItem('auth_token');
        const email = localStorage.getItem('user_email');
        if (token && email) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {token: token, email: email}
            }, '*');
        }
    </script>
    """
    
    # We'll handle this in the main app with a custom component
    # For now, return False - we'll implement this properly in next step
    return False
