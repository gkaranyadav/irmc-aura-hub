import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration - FIXED for full display
st.set_page_config(
    page_title="IRMC Aura - AI Apps Hub",
    page_icon="⚡",
    layout="centered",  # Changed from "wide" to prevent cutting
    initial_sidebar_state="collapsed"
)

# Custom CSS - COMPLETELY FIXED
st.markdown("""
<style>
    /* FIX: Remove all default padding and margins */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        padding: 0;
        line-height: 1.1;
    }
    
    .tagline {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin: 0 0 2rem 0;
        padding: 0;
        font-weight: 400;
    }
    
    /* FIX: Remove white background from main container */
    .main-container {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* FIX: Make login container transparent background */
    .login-tabs-container {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 0 auto;
        max-width: 500px;
        box-shadow: 0 10px 40px rgba(23, 92, 255, 0.15);
        border: 1px solid #E6F0FF;
        backdrop-filter: blur(10px);
    }
    
    /* FIX: Remove Streamlit default padding */
    .stApp {
        background: linear-gradient(135deg, #F8FAFF 0%, #FFFFFF 100%) !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* FIX: Style form elements properly */
    .stTextInput input, .stTextInput input:focus {
        border: 2px solid #E6F0FF !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        background: #FFFFFF !important;
    }
    
    .stTextInput input:focus {
        border-color: #175CFF !important;
        box-shadow: 0 0 0 1px #175CFF !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #175CFF, #00A3FF) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(23, 92, 255, 0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #F8FAFF;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #175CFF !important;
        color: white !important;
    }
    
    /* Remove extra spaces */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Hide header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def login_page():
    # FIXED: Clean header without getting cut
    st.markdown("""
    <div class="main-container">
        <div class="main-header">⚡ IRMC Aura</div>
        <div class="tagline">Intelligent AI Applications Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # FIXED: Login container with semi-transparent background
    st.markdown('<div class="login-tabs-container">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["**🔐 Login**", "**✨ Sign Up**"])
    
    with tab1:
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 1.5rem;">Welcome Back</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("**Email Address**", placeholder="your.email@example.com")
            password = st.text_input("**Password**", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("**🚀 Login to Dashboard**")
            
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
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 1.5rem;">Create Account</h3>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            new_email = st.text_input("**Email Address**", placeholder="your.email@example.com")
            new_password = st.text_input("**Password**", type="password", placeholder="Create a password (min 4 chars)")
            confirm_password = st.text_input("**Confirm Password**", type="password", placeholder="Re-enter your password")
            signup_btn = st.form_submit_button("**⭐ Create Account**")
            
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
    # FIXED: Homepage header
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("""
        <div style='margin-bottom: 1rem;'>
            <div class="main-header" style='font-size: 2.5rem; text-align: left; margin: 0;'>⚡ IRMC Aura</div>
            <p style='color: #666; font-size: 1.1rem; margin: 0;'>
                Welcome back, <strong style="color: #175CFF;">{}</strong>
            </p>
        </div>
        """.format(st.session_state.email), unsafe_allow_html=True)
    
    with col2:
        if st.button("**🚪 Logout**", use_container_width=True):
            logout_user()
            st.rerun()
    
    st.markdown("---")
    
    # App boxes
    st.markdown('<h3 style="color: #175CFF; text-align: center;">Your AI Applications</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF; box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>📚 Document RAG Chat</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>Chat with your documents using advanced AI technology.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open RAG Chat", key="rag_btn", use_container_width=True):
            st.switch_page("pages/rag_chat.py")
        
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF; box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🌐 GraphDB RAG</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>Advanced document analysis with knowledge graphs.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF; box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>💾 SQL Generator</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>Convert English to SQL queries instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open SQL Generator", key="sql_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
        
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF; box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🎯 Future Apps</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>New AI tools coming soon.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore", key="new_btn", use_container_width=True):
            st.info("🎯 New features launching soon!")

def main():
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
