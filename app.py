import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="IRMC Aura - AI Apps Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
    }
    .app-box {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .app-box:hover {
        border-color: #1E3A8A;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1E40AF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # IRMC Aura Header
    st.markdown('<div class="main-header">🔮 IRMC Aura</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280; font-size: 1.2rem; margin-bottom: 2rem;">Your Gateway to AI Applications</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚀 **Login**", "📝 **Sign Up**"])
    
    with tab1:
        with st.form("login_form"):
            st.markdown('<div class="sub-header">Welcome Back</div>', unsafe_allow_html=True)
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("🚀 Login to IRMC Aura")
            
            if login_btn:
                if email and password:
                    if login_user(email, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
                else:
                    st.warning("⚠️ Please fill all fields")
    
    with tab2:
        with st.form("signup_form"):
            st.markdown('<div class="sub-header">Join IRMC Aura</div>', unsafe_allow_html=True)
            new_email = st.text_input("📧 New Email Address", placeholder="Enter your email")
            new_password = st.text_input("🔒 New Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            signup_btn = st.form_submit_button("✨ Create Account")
            
            if signup_btn:
                if new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("❌ Passwords don't match!")
                    elif len(new_password) < 4:
                        st.error("❌ Password must be at least 4 characters")
                    else:
                        if signup_user(new_email, new_password):
                            st.success("🎉 Account created successfully! Please login.")
                        else:
                            st.error("❌ Email already exists!")
                else:
                    st.warning("⚠️ Please fill all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

def home_page():
    # Header with logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🔮 IRMC Aura</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #6B7280; font-size: 1.2rem;">Welcome back, <strong>{st.session_state.email}</strong>! 👋</p>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
    
    st.markdown("---")
    
    st.markdown('<div class="sub-header" style="text-align: center; margin-bottom: 2rem;">🎯 Your AI Applications</div>', unsafe_allow_html=True)
    
    # App boxes in 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # App 1: Document RAG Chat
        st.markdown("""
        <div class="app-box">
            <h3>📄 Document RAG Chat</h3>
            <p>Chat with your documents using advanced AI and Retrieval-Augmented Generation</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open RAG Chat", key="rag_btn", use_container_width=True):
            st.switch_page("pages/rag_chat.py")
        
        # App 2: GraphDB RAG
        st.markdown("""
        <div class="app-box">
            <h3>🕸️ GraphDB RAG</h3>
            <p>Advanced document analysis with knowledge graphs and semantic relationships</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
    
    with col2:
        # App 3: SQL Query Generator
        st.markdown("""
        <div class="app-box">
            <h3>🗃️ SQL Query Generator</h3>
            <p>Convert natural English language to optimized SQL queries instantly</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open SQL Generator", key="sql_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
        
        # App 4: New App
        st.markdown("""
        <div class="app-box">
            <h3>➕ New Application</h3>
            <p>Your next innovative AI tool - Stay tuned for updates!</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore", key="new_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")

def main():
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
