import sqlite3
import bcrypt
from datetime import datetime

def init_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_credentials (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Hash password
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # Generate user ID
    user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        cursor.execute('''
            INSERT INTO user_credentials (user_id, email, password_hash)
            VALUES (?, ?, ?)
        ''', (user_id, email, password_hash))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, password_hash FROM user_credentials 
        WHERE email = ? AND is_active = TRUE
    ''', (email,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result and bcrypt.checkpw(password.encode(), result[1].encode()):
        return result[0]  # Return user_id
    return None
