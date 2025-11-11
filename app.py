import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="IRMC Aura - AI Apps Hub",
    page_icon="🌊",  # Wave icon for sea theme
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for deep sea blue theme
st.markdown("""
<style>
    /* Remove white box and fix styling */
    .main-header {
        font-size: 3.5rem;
        color: #1e3a8a;
        text-align: center;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #1e3a8a, #3730a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(30, 58, 138, 0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #475569;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .app-box {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    .app-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(30, 58, 138, 0.15);
        border-left: 5px solid #3730a3;
    }
    .login-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        max-width: 500px;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.1);
        border: 1px solid #e2e8f0;
    }
    .stButton button {
        background: linear-gradient(135deg, #1e3a8a, #3730a3);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #3730a3, #1e3a8a);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        gap: 1rem;
        padding: 0 1rem;
    }
    /* Remove any extra white spaces */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style the tabs better */
    [data-testid="stHorizontalBlock"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

def login_page():
    # Main container without extra white box
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <div class="main-header">🌊 IRMC Aura</div>
        <p style="color: #64748b; font-size: 1.3rem; font-weight: 500;">Your Gateway to Intelligent AI Applications</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login/Signup container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚀 **Login to Dashboard**", "✨ **Create New Account**"])
    
    with tab1:
        st.markdown('<div class="sub-header">Welcome Back</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            email = st.text_input("📧 **Email Address**", placeholder="your.email@example.com")
            password = st.text_input("🔒 **Password**", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("🌊 **Access IRMC Aura**")
            
            if login_btn:
                if email and password:
                    if login_user(email, password):
                        st.success("🎉 Welcome back! Redirecting to your dashboard...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password. Please try again.")
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    with tab2:
        st.markdown('<div class="sub-header">Join IRMC Aura</div>', unsafe_allow_html=True)
        with st.form("signup_form"):
            new_email = st.text_input("📧 **Email Address**", placeholder="your.email@example.com")
            new_password = st.text_input("🔒 **Create Password**", type="password", placeholder="Minimum 4 characters")
            confirm_password = st.text_input("🔒 **Confirm Password**", type="password", placeholder="Re-enter your password")
            signup_btn = st.form_submit_button("✨ **Start Your Journey**")
            
            if signup_btn:
                if new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("❌ Passwords don't match! Please try again.")
                    elif len(new_password) < 4:
                        st.error("❌ Password must be at least 4 characters long.")
                    else:
                        if signup_user(new_email, new_password):
                            st.success("🎉 Account created successfully! Please login to continue.")
                        else:
                            st.error("❌ This email is already registered. Please use a different email.")
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

def home_page():
    # Header section
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        <div style='text-align: left;'>
            <div class="main-header" style='text-align: left; font-size: 2.5rem;'>🌊 IRMC Aura</div>
            <p style="color: #64748b; font-size: 1.1rem; margin-top: -0.5rem;">Welcome back, <strong style="color: #1e3a8a;">{}</strong>! 👋</p>
        </div>
        """.format(st.session_state.email), unsafe_allow_html=True)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 **Logout**", use_container_width=True):
            logout_user()
            st.rerun()
    
    st.markdown("---")
    
    # Dashboard title
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h2 style='color: #1e3a8a; font-size: 2rem; font-weight: 700;'>Your AI Applications Dashboard</h2>
        <p style='color: #64748b; font-size: 1.1rem;'>Choose from our suite of intelligent AI tools</p>
    </div>
    """, unsafe_allow_html=True)
    
    # App boxes in 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # App 1: Document RAG Chat
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #1e3a8a; margin-bottom: 1rem;'>📄 Document RAG Chat</h3>
            <p style='color: #475569; line-height: 1.6;'>Chat with your documents using advanced AI and Retrieval-Augmented Generation technology. Ask questions and get intelligent answers from your documents.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch RAG Chat", key="rag_btn", use_container_width=True):
            st.switch_page("pages/rag_chat.py")
        
        # App 2: GraphDB RAG
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #1e3a8a; margin-bottom: 1rem;'>🕸️ GraphDB RAG</h3>
            <p style='color: #475569; line-height: 1.6;'>Advanced document analysis with knowledge graphs and semantic relationships. Discover connections and insights in your data.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Explore Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Feature coming soon! Stay tuned.")
    
    with col2:
        # App 3: SQL Query Generator
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #1e3a8a; margin-bottom: 1rem;'>🗃️ SQL Query Generator</h3>
            <p style='color: #475569; line-height: 1.6;'>Convert natural English language to optimized SQL queries instantly. No SQL knowledge required - just ask what you need!</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Generate SQL", key="sql_btn", use_container_width=True):
            st.info("🚧 Feature coming soon! We're working on it.")
        
        # App 4: New App
        st.markdown("""
        <div class="app-box">
            <h3 style='color: #1e3a8a; margin-bottom: 1rem;'>🔮 Future Innovations</h3>
            <p style='color: #475569; line-height: 1.6;'>We're constantly developing new AI tools. Stay connected for the latest innovations and updates to the IRMC Aura platform.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🌟 Coming Soon", key="new_btn", use_container_width=True):
            st.info("🎯 Exciting new features are on the way!")

def main():
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
