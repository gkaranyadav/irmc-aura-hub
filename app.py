import streamlit as st
import extra_streamlit_components as stx
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Initialize cookie manager
@st.cache_resource
def get_cookie_manager():
    return stx.CookieManager()

cookie_manager = get_cookie_manager()

# Page configuration
st.set_page_config(
    page_title="IRMC aura - AI Apps Hub",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
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
    
    .app-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1);
        margin-bottom: 1rem;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
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
    
    div[data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(23, 92, 255, 0.15);
        border: 1px solid #E6F0FF;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="main-header">🔮 IRMC aura</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["**🔐 Login**", "**✨ Sign Up**"])
    
    with tab1:
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Welcome Back</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("**Email Address**", placeholder="test@test.com")
            password = st.text_input("**Password**", type="password", placeholder="1234")
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
            new_password = st.text_input("**Password**", type="password", placeholder="Create a password")
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
    # Homepage header
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
    
    # App boxes
    st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Your AI Applications</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>📄 Doc RAG Chat</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Chat with your documents using advanced AI technology.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch RAG Chat", key="rag_btn", use_container_width=True):
            st.info("🚧 Coming Soon - RAG Chat integration in progress!")
        
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🌐 Graph RAG</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Advanced document analysis with knowledge graphs.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")
    
    with col2:
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
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Check authentication (session OR cookie)
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
