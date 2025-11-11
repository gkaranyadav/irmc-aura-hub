import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="IRMC Aura - AI Apps Hub",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# ✅ Custom CSS - Cleaned and Fixed White Box Issue
st.markdown("""
<style>
    /* Clean header without tagline */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin: 1rem 0 2rem 0;
        padding: 0;
        line-height: 1.1;
    }
    /* Clean login container */
    .login-tabs-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 0.5rem;
        margin: 0 auto;
        max-width: 500px;
        box-shadow: 0 10px 40px rgba(23, 92, 255, 0.15);
        border: 3px solid #E6F0FF;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #F8FAFF 0%, #FFFFFF 100%);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Buttons */
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

    /* Tabs container */
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
        background-color: #175CFF;
        color: white;
    }

    /* Hide Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ✅ FIX for white box under header */
    div[data-testid="stTabs"] {
        background: transparent !important;
        box-shadow: none !important;
        margin-top: -1rem !important;
        padding-top: 0 !important;
    }

    /* Make sure container below header aligns properly */
    .login-tabs-container {
        margin-top: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- LOGIN PAGE --------------------
def login_page():
    st.markdown('<div class="main-header">⚡ IRMC Aura</div>', unsafe_allow_html=True)

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

# -------------------- HOME PAGE --------------------
def home_page():
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"""
        <div style='margin-bottom: 1rem;'>
            <div class="main-header" style='font-size: 2.5rem; text-align: left; margin: 0;'>⚡ IRMC Aura</div>
            <p style='color: #666; font-size: 1.1rem; margin: 0;'>
                Welcome back, <strong style="color: #175CFF;">{st.session_state.email}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("**🚪 Logout**", use_container_width=True):
            logout_user()
            st.rerun()

    st.markdown("---")

    st.markdown('<h3 style="color: #175CFF; text-align: center;">Your AI Applications</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>📚 Document RAG Chat</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>Chat with your documents using advanced AI technology.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open RAG Chat", key="rag_btn", use_container_width=True):
            st.switch_page("pages/rag_chat.py")

        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🌐 GraphDB RAG</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>Advanced document analysis with knowledge graphs.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Graph RAG", key="graph_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")

    with col2:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>💾 SQL Generator</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>Convert English to SQL queries instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open SQL Generator", key="sql_btn", use_container_width=True):
            st.info("🚧 Coming Soon!")

        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1); margin-bottom: 1rem;'>
            <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🎯 Future Apps</h4>
            <p style='color: #555; margin: 0; font-size: 0.9rem;'>New AI tools coming soon.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore", key="new_btn", use_container_width=True):
            st.info("🎯 New features launching soon!")

# -------------------- MAIN --------------------
def main():
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
