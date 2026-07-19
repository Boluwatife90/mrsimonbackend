import sqlite3
import os

DB_PATH = 'pfdss_users.db'

conn = sqlite3.connect(DB_PATH)

# Insert seasonal data
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
    conn.execute('''INSERT OR IGNORE INTO seasonal_calendar 
        (crop_type, region, planting_start, planting_end, harvest_start, harvest_end) 
        VALUES (?,?,?,?,?,?)''', s)

conn.commit()
conn.close()

print("✅ Seasonal data added successfully!")