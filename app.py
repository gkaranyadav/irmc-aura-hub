import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="IRMC Aura - AI Apps Hub",
    page_icon="⚡",  # Stylish bolt icon
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with DeepSeek blue theme
st.markdown("""
<style>
    /* Remove all padding/margin issues */
    .main-header {
        font-size: 4rem;
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }
    
    .tagline {
        font-size: 1.4rem;
        color: #666;
        text-align: center;
        margin: 0.5rem 0 3rem 0;
        font-weight: 500;
    }
    
    /* Main container without white box */
    .main-container {
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    .login-tabs-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin: 0 auto;
        max-width: 500px;
        box-shadow: 0 10px 40px rgba(23, 92, 255, 0.15);
        border: 1px solid #E6F0FF;
    }
    
    .app-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #F0F5FF;
        box-shadow: 0 4px 20px rgba(23, 92, 255, 0.08);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .app-box:hover {
        transform: translateY(-5px);
        border-color: #175CFF;
        box-shadow: 0 8px 30px rgba(23, 92, 255, 0.15);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 92, 255, 0.3);
    }
    
    /* Fix tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #F8FAFF;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #175CFF !important;
        color: white !important;
    }
    
    /* Remove any extra white spaces */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0;
    }
</style>
""", unsafe_allow_html=True)

def login_page():
    # Clean header without white boxes
    st.markdown("""
    <div class="main-container">
        <div class="main-header">⚡ IRMC Aura</div>
        <div class="tagline">Intelligent AI Applications Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login/Signup tabs
    st.markdown('<div class="login-tabs-container">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 **Login**", "✨ **Sign Up**"])
    
    with tab1:
        st.markdown('<h3 style="color: #175CFF; margin-bottom: 1.5rem;">Welcome Back</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("**Email Address**", placeholder="Enter your email")
            password = st.text_input("**Password**", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("🚀 **Login to Dashboard**")
            
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
        st.markdown('<h3 style="color: #175CFF; margin-bottom: 1.5rem;">Join IRMC Aura</h3>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            new_email = st.text_input("**Email Address**", placeholder="Enter your email")
            new_password = st.text_input("**Password**", type="password", placeholder="Create a password")
            confirm_password = st.text_input("**Confirm Password**", type="password", placeholder="Confirm your password")
            signup_btn = st.form_submit_button("⭐ **Create Account**")
            
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
    # Clean header section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <div class="main-header" style='font-size: 3rem; text-align: left;'>⚡ IRMC Aura</div>
            <p style='color: #666; font-size: 1.2rem; margin-top: -0.5rem;'>
                Welcome back, <strong style="color: #175CFF;">{}</strong>! Ready to create?
            </p>
        </div>
        """.format(st.session_state.email), unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 **Logout**", use_container_width=True):
            logout_user()
            st.rerun()
    
    st.markdown("---")
    
    # Dashboard title
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 3rem 0;'>
        <h2 style='color: #175CFF; font-size: 2.2rem; font-weight: 700;'>Your AI Workspace</h2>
        <p style='color: #666; font-size: 1.1rem;'>Choose your intelligent AI tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # App boxes
    col1, col2 = st.columns(2)
    
    with col1:
        # App 1
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #175CFF; margin-bottom: 1rem;'>📚 Document RAG Chat</h3>
            <p style='color: #555; line-height: 1.6;'>Chat with your documents using advanced AI. Ask questions and get intelligent answers instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch RAG Chat", key="rag_btn", use_container_width=True):
            st.switch_page("pages/rag_chat.py")
        
        # App 2
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #175CFF; margin-bottom: 1rem;'>🌐 GraphDB RAG</h3>
            <p style='color: #555; line-height: 1.6;'>Advanced document analysis with knowledge graphs and semantic relationships.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Explore Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
    
    with col2:
        # App 3
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #175CFF; margin-bottom: 1rem;'>💾 SQL Query Generator</h3>
            <p style='color: #555; line-height: 1.6;'>Convert natural English to optimized SQL queries. No coding knowledge required.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Generate SQL", key="sql_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
        
        # App 4
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #175CFF; margin-bottom: 1rem;'>🎯 Future Innovations</h3>
            <p style='color: #555; line-height: 1.6;'>We're constantly developing new AI tools to enhance your productivity.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🌟 Coming Soon", key="new_btn", use_container_width=True):
            st.info("🎯 New features launching soon!")

def main():
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
