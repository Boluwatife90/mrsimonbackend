import sqlite3
from werkzeug.security import generate_password_hash

# --- CONFIGURATION ---
DB_PATH = 'pfdss_users.db'
NEW_PASSWORD = 'simple_simon@1234'
YOUR_EMAIL = 'simple4real08@gmail.com'  # <--- CHANGE THIS TO YOUR ACTUAL EMAIL

# --- UPDATE PASSWORD ---
try:
    conn = sqlite3.connect(DB_PATH)
    hashed_password = generate_password_hash(NEW_PASSWORD)
    
    # Update the password for your email
    cursor = conn.execute('UPDATE users SET password_hash = ? WHERE email = ?', (hashed_password, YOUR_EMAIL))
    
    if cursor.rowcount > 0:
        conn.commit()
        print(f"✅ SUCCESS! Password updated for {YOUR_EMAIL}")
        print(f"🔑 New Password: {NEW_PASSWORD}")
    else:
        print(f" ERROR: Email '{YOUR_EMAIL}' not found in the database.")
        print("Please check your spelling or look at the list of users below:")
        users = conn.execute('SELECT email, full_name FROM users').fetchall()
        for u in users:
            print(f" - {u[0]} ({u[1]})")
            
    conn.close()
except Exception as e:
    print(f"❌ An error occurred: {e}")

input("\nPress Enter to close...")