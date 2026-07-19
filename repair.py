import os

file_path = 'app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Fix the login token spaces (The #1 cause of immediate logout)
code = code.replace('"access_token "', '"access_token"')
code = code.replace('"user "', '"user"')
code = code.replace('"email "', '"email"')
code = code.replace('"name "', '"name"')
code = code.replace('"error "', '"error"')
code = code.replace('"message "', '"message"')
code = code.replace('"Success "', '"Success"')
code = code.replace('"Invalid credentials "', '"Invalid credentials"')

# 2. Fix the broken password checker (Causes 500 error on login)
code = code.replace('chec k_password_hash', 'check_password_hash')

# 3. Fix server startup errors
code = code.replace('Flask(name)', 'Flask(__name__)')
code = code.replace('abspath(file)', 'abspath(__file__)')
code = code.replace("if name == 'main':", "if __name__ == '__main__':")
code = code.replace('[int:user_id](int:user_id)', '<int:user_id>')

# 4. Fix comment syntax errors (Causes SyntaxError)
code = code.replace('-- coding: utf-8 --', '# -*- coding: utf-8 -*-')
code = code.replace('--- ML Model Loading ---', '# --- ML Model Loading ---')
code = code.replace('--- App Initialization ---', '# --- App Initialization ---')
code = code.replace('--- Frontend Routes ---', '# --- Frontend Routes ---')
code = code.replace('--- Auth Routes ---', '# --- Auth Routes ---')
code = code.replace('--- USER CONTROLS ---', '# --- USER CONTROLS ---')
code = code.replace('--- SYSTEM SETTINGS ---', '# --- SYSTEM SETTINGS ---')
code = code.replace('--- DATA CONTROL ---', '# --- DATA CONTROL ---')
code = code.replace('--- Helper Functions ---', '# --- Helper Functions ---')
code = code.replace('--- Main Prediction Route ---', '# --- Main Prediction Route ---')
code = code.replace('--- Feedback & Sensor Routes ---', '# --- Feedback & Sensor Routes ---')
code = code.replace('--- Start Server ---', '# --- Start Server ---')
code = code.replace('============================================', '# ============================================')
code = code.replace('✅ ADMIN CONTROLS', '# ✅ ADMIN CONTROLS')
code = code.replace('1. Define Decorator First', '# 1. Define Decorator First')
code = code.replace('2. Admin Routes', '# 2. Admin Routes')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("✅ app.py has been repaired! Restart your server now.")