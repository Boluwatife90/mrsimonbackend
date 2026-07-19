import sqlite3
import os

DB_PATH = 'pfdss_users.db'

print(f"🔍 Checking database at: {os.path.abspath(DB_PATH)}")

if not os.path.exists(DB_PATH):
    print("❌ Database file not found. Please run app.py first.")
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. Ensure Regional Table exists and add data
cursor.execute('''CREATE TABLE IF NOT EXISTS regional_defaults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_name TEXT UNIQUE NOT NULL,
    default_ph REAL,
    default_moisture REAL,
    typical_n REAL,
    typical_p REAL,
    typical_k REAL,
    climate_zone TEXT
)''')

default_regions = [
    ('Lagos', 6.0, 35, 25, 18, 180, 'Tropical Coastal'),
    ('Abuja', 6.5, 30, 20, 15, 150, 'Savanna'),
    ('Kano', 7.0, 25, 18, 12, 140, 'Sahel Savanna'),
    ('Ibadan', 6.2, 32, 22, 16, 170, 'Tropical Savanna'),
    ('Port Harcourt', 5.8, 38, 28, 20, 190, 'Tropical Coastal'),
    ('Benin City', 6.1, 34, 24, 17, 175, 'Tropical Rainforest'),
    ('Maiduguri', 7.2, 22, 15, 10, 130, 'Sahel'),
    ('Kaduna', 6.8, 28, 19, 14, 145, 'Savanna'),
    ('Enugu', 6.3, 33, 23, 16, 165, 'Tropical Rainforest')
]

for r in default_regions:
    cursor.execute('INSERT OR IGNORE INTO regional_defaults (region_name, default_ph, default_moisture, typical_n, typical_p, typical_k, climate_zone) VALUES (?,?,?,?,?,?,?)', r)
print("✅ Regional data updated.")

# 2. Ensure Seasonal Table exists and add data
cursor.execute('''CREATE TABLE IF NOT EXISTS seasonal_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_type TEXT NOT NULL,
    region TEXT NOT NULL,
    planting_start TEXT,
    planting_end TEXT,
    harvest_start TEXT,
    harvest_end TEXT,
    UNIQUE(crop_type, region)
)''')

default_seasons = [
    ('Maize', 'Lagos', 'March', 'April', 'July', 'August'),
    ('Maize', 'Abuja', 'April', 'May', 'August', 'September'),
    ('Maize', 'Kano', 'May', 'June', 'September', 'October'),
    ('Rice', 'Lagos', 'April', 'May', 'September', 'October'),
    ('Rice', 'Abuja', 'May', 'June', 'October', 'November'),
    ('Tomato', 'Lagos', 'October', 'November', 'February', 'March'),
    ('Tomato', 'Kano', 'September', 'October', 'January', 'February'),
    ('Cassava', 'Lagos', 'January', 'March', 'January', 'December'),
    ('Cassava', 'Abuja', 'February', 'April', 'February', 'December'),
    ('Pepper', 'Lagos', 'September', 'October', 'January', 'February'),
    ('Pepper', 'Kano', 'August', 'September', 'December', 'January')
]

for s in default_seasons:
    cursor.execute('INSERT OR IGNORE INTO seasonal_calendar (crop_type, region, planting_start, planting_end, harvest_start, harvest_end) VALUES (?,?,?,?,?,?)', s)
print("✅ Seasonal data updated.")

conn.commit()
conn.close()
print("🎉 SUCCESS! Database fixed without deleting any users.")