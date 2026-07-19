import sqlite3

DB_PATH = 'pfdss_users.db'

print("🧹 Starting duplicate removal for seasonal calendar...")

conn = sqlite3.connect(DB_PATH)

# Count before cleanup
before_count = conn.execute('SELECT COUNT(*) FROM seasonal_calendar').fetchone()[0]
print(f"📊 Total records before: {before_count}")

# Show duplicates
duplicates = conn.execute('''
    SELECT crop_type, region, state, COUNT(*) as count
    FROM seasonal_calendar
    GROUP BY crop_type, region, state
    HAVING count > 1
''').fetchall()

if duplicates:
    print(f"\n⚠️ Found {len(duplicates)} duplicate entries:")
    for dup in duplicates:
        print(f"   - {dup[0]} in {dup[1]} ({dup[2]}): {dup[3]} copies")
    
    # Delete duplicates (keep only the first occurrence)
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
    deleted = before_count - after_count
    
    print(f"\n✅ Cleanup complete!")
    print(f"📊 Total records after: {after_count}")
    print(f"🗑️ Removed {deleted} duplicate entries")
else:
    print("✅ No duplicates found. Database is clean!")

conn.close()
print("\n🎉 Done!")