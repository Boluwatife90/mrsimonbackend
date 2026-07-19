import sqlite3

DB_PATH = 'pfdss_users.db'

print("🧹 Starting seasonal data cleanup...")

conn = sqlite3.connect(DB_PATH)

# Count before cleanup
before_count = conn.execute('SELECT COUNT(*) FROM seasonal_calendar').fetchone()[0]
print(f"📊 Total records before cleanup: {before_count}")

# Count duplicates
duplicate_count = conn.execute('''
    SELECT COUNT(*) FROM (
        SELECT id FROM seasonal_calendar
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM seasonal_calendar 
            GROUP BY crop_type, region, state
        )
    )
''').fetchone()[0]
print(f"⚠️ Found {duplicate_count} duplicate entries")

if duplicate_count > 0:
    # Delete duplicates (keep only the first occurrence of each)
    conn.execute('''
        DELETE FROM seasonal_calendar 
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM seasonal_calendar 
            GROUP BY crop_type, region, state
        )
    ''')
    conn.commit()
    
    # Count after cleanup
    after_count = conn.execute('SELECT COUNT(*) FROM seasonal_calendar').fetchone()[0]
    print(f"✅ Cleanup complete!")
    print(f"📊 Total records after cleanup: {after_count}")
    print(f"🗑️ Removed {before_count - after_count} duplicate entries")
else:
    print("✅ No duplicates found. Database is clean!")

conn.close()
print("\n🎉 Done! You can now restart your Flask server.")