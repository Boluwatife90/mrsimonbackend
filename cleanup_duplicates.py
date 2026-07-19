import sqlite3

DB_PATH = 'pfdss_users.db'

print("🧹 Starting database cleanup...")

conn = sqlite3.connect(DB_PATH)

# 1. Remove duplicate seasonal entries (keep only the first occurrence)
print("\n📅 Cleaning seasonal_calendar table...")
before_count = conn.execute('SELECT COUNT(*) FROM seasonal_calendar').fetchone()[0]
print(f"   Before cleanup: {before_count} records")

conn.execute('''
    DELETE FROM seasonal_calendar 
    WHERE id NOT IN (
        SELECT MIN(id) 
        FROM seasonal_calendar 
        GROUP BY crop_type, region, state
    )
''')

after_count = conn.execute('SELECT COUNT(*) FROM seasonal_calendar').fetchone()[0]
deleted = before_count - after_count
print(f"   After cleanup: {after_count} records")
print(f"   ✅ Removed {deleted} duplicate entries")

# 2. Verify the data is clean
print("\n🔍 Verification - Sample records:")
samples = conn.execute('''
    SELECT crop_type, region, state, planting_start, planting_end 
    FROM seasonal_calendar 
    ORDER BY crop_type, region 
    LIMIT 10
''').fetchall()

for s in samples:
    print(f"   {s[0]} in {s[1]} ({s[2]}): {s[3]} - {s[4]}")

conn.commit()
conn.close()

print("\n✅ Cleanup complete! Your database is now clean.")
print("   Restart your Flask server to apply changes.")