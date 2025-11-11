import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="IRMC aura - AI Apps Hub",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple CSS
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
    }
</style>
""", unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="main-header">🔮 IRMC aura</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["**🔐 Login**", "**✨ Sign Up**"])
    
    with tab1:
        st.markdown('<h3 style="color: #175CFF; text-align: center;">Welcome Back</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if login_user(email, password):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Login failed")
    
    with tab2:
        st.markdown('<h3 style="color: #175CFF; text-align: center;">Create Account</h3>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Sign Up"):
                if password == confirm:
                    if signup_user(email, password):
                        st.success("✅ Account created!")
                    else:
                        st.error("❌ Email exists")
                else:
                    st.error("❌ Passwords don't match")

def home_page():
    st.markdown('<div class="main-header">🔮 IRMC aura</div>', unsafe_allow_html=True)
    st.write(f"Welcome, {st.session_state.email}!")
    
    if st.button("Logout"):
        logout_user()
        st.rerun()

def main():
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
