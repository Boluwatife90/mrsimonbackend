import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = 'pfdss_users.db'

# Check if DB exists
if not os.path.exists(DB_PATH):
    print(" Error: 'pfdss_users.db' not found. Please run 'python app.py' first!")
    exit()

conn = sqlite3.connect(DB_PATH)

# --- YOUR ADMIN DETAILS ---
email = 'simple4real08@gmail.com'
name = 'simple'
password = 'admin123'  # You can change this password
role = 'admin'
# --------------------------

password_hash = generate_password_hash(password)

try:
    conn.execute('''INSERT INTO users (email, password_hash, full_name, role) 
                    VALUES (?, ?, ?, ?)''', 
                 (email, password_hash, name, role))
    conn.commit()
    print(f"✅ SUCCESS! Admin account created.")
    print(f"👤 Name: {name}")
    print(f" Email: {email}")
    print(f" Password: {password}")
except sqlite3.IntegrityError:
    print("⚠️ User already exists. Updating role to 'admin'...")
    conn.execute("UPDATE users SET role = 'admin' WHERE email = ?", (email,))
    conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (password_hash, email))
    conn.commit()
    print("✅ Account restored as Admin.")

conn.close()
print("\n🚀 You can now run 'python app.py' and login!")