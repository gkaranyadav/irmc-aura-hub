import streamlit as st
import jwt
import datetime

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
    """Login user and set session + local storage"""
    from database import verify_user
    user_id = verify_user(email, password)
    if user_id:
        token = generate_token(user_id, email)
        
        # Store in session state
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.email = email
        st.session_state.token = token
        
        # Store in local storage
        store_token_in_local_storage(token, email)
        return True
    return False

def signup_user(email, password):
    """Create new user account"""
    from database import create_user
    return create_user(email, password)

def logout_user():
    """Logout user and clear all storage"""
    for key in ['logged_in', 'user_id', 'email', 'token']:
        if key in st.session_state:
            del st.session_state[key]
    clear_local_storage()

def check_session():
    """Check if user is logged in (session + local storage)"""
    # First check session state (for same session)
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        try:
            jwt.decode(st.session_state.token, SECRET_KEY, algorithms=['HS256'])
            return True
        except jwt.ExpiredSignatureError:
            logout_user()
            return False
    
    # If no session, check local storage (for page refresh/new tab)
    return check_local_storage_token()

def store_token_in_local_storage(token, email):
    """Store token in browser local storage"""
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
    """Check if valid token exists in local storage and auto-login"""
    # Use a more reliable approach - check query parameters
    query_params = st.experimental_get_query_params()
    
    # Check if we have token data from a previous successful login
    if 'auth_token' in st.session_state and 'user_email' in st.session_state:
        token = st.session_state.auth_token
        email = st.session_state.user_email
        
        # Verify the token
        payload = verify_token(token)
        if payload:
            # Token is valid - auto-login the user
            st.session_state.logged_in = True
            st.session_state.user_id = payload['user_id']
            st.session_state.email = payload['email']
            st.session_state.token = token
            return True
    
    # Simple approach: Try to read from localStorage via JavaScript and set in session
    if 'local_storage_check_done' not in st.session_state:
        st.session_state.local_storage_check_done = True
        
        # JavaScript to check localStorage and set a flag
        js_code = """
        <script>
            const token = localStorage.getItem('auth_token');
            const email = localStorage.getItem('user_email');
            if (token && email) {
                // Set a URL parameter that Streamlit can read
                window.location.href = window.location.pathname + '?auth_token=' + encodeURIComponent(token) + '&auth_email=' + encodeURIComponent(email);
            }
        </script>
        """
        st.components.v1.html(js_code, height=0)
    
    # Check URL parameters for auth data (set by JavaScript above)
    if 'auth_token' in query_params and 'auth_email' in query_params:
        token = query_params['auth_token'][0]
        email = query_params['auth_email'][0]
        
        # Verify the token
        payload = verify_token(token)
        if payload:
            # Token is valid - auto-login the user
            st.session_state.logged_in = True
            st.session_state.user_id = payload['user_id']
            st.session_state.email = payload['email']
            st.session_state.token = token
            st.session_state.auth_token = token
            st.session_state.user_email = email
            
            # Clear the URL parameters
            st.experimental_set_query_params()
            return True
        else:
            # Token is invalid - clear local storage
            clear_local_storage()
    
    return False
# Function to handle messages from JavaScript
def handle_auth_message(data):
    """Store auth data from JavaScript for processing"""
    if 'token' in data and 'email' in data:
        st.session_state.auth_token_data = {
            'token': data['token'],
            'email': data['email']
        }
