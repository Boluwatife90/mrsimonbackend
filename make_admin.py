import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pfdss_users.db')
YOUR_EMAIL = 'simple4real08@gmail.com'  # Make sure this is your email

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

user = conn.execute('SELECT * FROM users WHERE email = ?', (YOUR_EMAIL,)).fetchone()

if not user:
    print(f"❌ Error: User with email '{YOUR_EMAIL}' not found!")
    conn.close()
    exit()

conn.execute("UPDATE users SET role = 'admin' WHERE email = ?", (YOUR_EMAIL,))
conn.commit()

user = conn.execute('SELECT * FROM users WHERE email = ?', (YOUR_EMAIL,)).fetchone()
conn.close()

print(f"✅ SUCCESS!")
print(f"User: {user['full_name']} ({user['email']})")
print(f"Role: {user['role'].upper()}")