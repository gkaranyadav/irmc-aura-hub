import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="IRMC aura - AI Apps Hub",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean CSS - FIXED ALIGNMENT
st.markdown("""
<style>
    /* Clean header */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin: 2rem 0 3rem 0;
        padding: 0;
    }
    
    /* Remove all default padding */
    .stApp {
        background: linear-gradient(135deg, #F8FAFF 0%, #FFFFFF 100%);
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Style form elements directly */
    div[data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(23, 92, 255, 0.15);
        border: 1px solid #E6F0FF;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 92, 255, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #F8FAFF;
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #175CFF;
        color: white;
    }
    
    /* FIXED: App box alignment */
    .app-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1);
        margin-bottom: 1rem;
        height: 140px; /* Fixed height for equal boxes */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Ensure columns have same spacing */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Hide header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def login_page():
    # CLEAN: Only the main header - FIXED NAME
    st.markdown('<div class="main-header"> IRMC aura</div>', unsafe_allow_html=True)
    
    # DIRECT tabs and forms - NO CONTAINERS
    tab1, tab2 = st.tabs(["**🔐 Login**", "**✨ Sign Up**"])
    
    with tab1:
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Welcome Back</h3>', unsafe_allow_html=True)
        
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
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Create Account</h3>', unsafe_allow_html=True)
        
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

def home_page():
    # Homepage header - FIXED NAME
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("""
        <div style='margin-bottom: 1rem;'>
            <div class="main-header" style='font-size: 2.5rem; text-align: left; margin: 0;'>🔮 IRMC aura</div>
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
    
    # App boxes - FIXED ALIGNMENT
    st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Your AI Applications</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # App 1: Document RAG Chat
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>📄 Document RAG Chat</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Chat with your documents using advanced AI technology.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch RAG Chat", key="rag_btn", use_container_width=True):
            st.switch_page("pages/rag_chat.py")
        
        # App 2: Graph RAG
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🌐 GraphDB RAG</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Advanced document analysis with knowledge graphs.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
    
    with col2:
        # App 3: SQL Generator
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>💬 English to SQL</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Convert English to SQL queries instantly.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate SQL", key="sql_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
        
        # App 4: Future Apps
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🚀 Future Apps</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>New AI tools coming soon.</p>
            </div>
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
