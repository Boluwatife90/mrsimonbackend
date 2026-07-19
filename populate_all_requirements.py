import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pfdss_users.db')

# ============================================
# 1. BASE NUTRIENT REQUIREMENTS PER CROP
# Format: (crop_name, vegetative_NPK, flowering_NPK, fruiting_NPK, pH_min, pH_max)
# ============================================
CROP_BASE_REQUIREMENTS = {
    # CEREALS
    'Maize': {
        'vegetative': (120, 60, 80), 'flowering': (150, 80, 100), 'fruiting': (100, 70, 90),
        'ph_min': 5.5, 'ph_max': 7.0, 'season': 'March-April'
    },
    'Rice': {
        'vegetative': (100, 50, 70), 'flowering': (130, 65, 85), 'fruiting': (90, 60, 80),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'April-May'
    },
    'Sorghum': {
        'vegetative': (80, 40, 60), 'flowering': (100, 50, 75), 'fruiting': (70, 35, 55),
        'ph_min': 6.0, 'ph_max': 7.5, 'season': 'May-June'
    },
    'Millet': {
        'vegetative': (60, 30, 50), 'flowering': (75, 38, 62), 'fruiting': (50, 25, 42),
        'ph_min': 6.0, 'ph_max': 7.5, 'season': 'June-July'
    },
    'Wheat': {
        'vegetative': (90, 50, 70), 'flowering': (110, 60, 85), 'fruiting': (85, 45, 65),
        'ph_min': 6.0, 'ph_max': 7.5, 'season': 'November-December'
    },
    
    # LEGUMES
    'Groundnut': {
        'vegetative': (50, 60, 90), 'flowering': (65, 75, 110), 'fruiting': (45, 55, 85),
        'ph_min': 5.8, 'ph_max': 7.0, 'season': 'June-July'
    },
    'Cowpea': {
        'vegetative': (40, 50, 70), 'flowering': (55, 65, 85), 'fruiting': (35, 45, 65),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'June-July'
    },
    'Soybean': {
        'vegetative': (60, 70, 80), 'flowering': (75, 85, 95), 'fruiting': (55, 65, 75),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'May-June'
    },
    
    # TUBERS & ROOTS
    'Yam': {
        'vegetative': (90, 45, 110), 'maturity': (110, 55, 130),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'February-April'
    },
    'Cassava': {
        'vegetative': (80, 40, 100), 'maturity': (60, 30, 80),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'January-March'
    },
    'Sweet Potato': {
        'vegetative': (70, 50, 100), 'tuber_bulking': (85, 60, 115), 'maturity': (65, 45, 95),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    'Cocoyam': {
        'vegetative': (85, 45, 105), 'corm_development': (100, 55, 120),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    'Irish Potato': {
        'vegetative': (120, 75, 130), 'tuber_initiation': (140, 85, 150), 'tuber_bulking': (115, 70, 125),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    
    # VEGETABLES
    'Tomato': {
        'vegetative': (150, 80, 120), 'flowering': (180, 100, 150), 'fruiting': (140, 90, 130),
        'ph_min': 6.0, 'ph_max': 6.8, 'season': 'October-November'
    },
    'Pepper': {
        'vegetative': (110, 60, 90), 'flowering': (140, 80, 110), 'fruiting': (100, 70, 100),
        'ph_min': 6.0, 'ph_max': 6.8, 'season': 'September-October'
    },
    'Okra': {
        'vegetative': (95, 55, 85), 'flowering': (110, 65, 100), 'fruiting': (90, 50, 80),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'March-May'
    },
    'Cabbage': {
        'vegetative': (140, 70, 110), 'head_formation': (160, 80, 130),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'September-November'
    },
    'Onion': {
        'vegetative': (110, 65, 95), 'bulb_formation': (130, 75, 115), 'maturity': (105, 60, 90),
        'ph_min': 6.0, 'ph_max': 7.5, 'season': 'October-December'
    },
    'Carrot': {
        'vegetative': (85, 60, 95), 'root_development': (100, 70, 110),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'September-November'
    },
    'Cucumber': {
        'vegetative': (90, 55, 85), 'flowering': (105, 65, 100), 'fruiting': (85, 50, 80),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'March-May'
    },
    'Lettuce': {
        'vegetative': (100, 50, 80), 'head_formation': (115, 60, 95),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'September-November'
    },
    'Melon': {
        'vegetative': (75, 55, 85), 'flowering': (90, 65, 100), 'fruiting': (70, 50, 80),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'March-May'
    },
    'Watermelon': {
        'vegetative': (80, 60, 90), 'flowering': (95, 70, 105), 'fruiting': (75, 55, 85),
        'ph_min': 6.0, 'ph_max': 7.0, 'season': 'March-May'
    },
    
    # FRUITS
    'Plantain': {
        'vegetative': (130, 70, 150), 'flowering': (150, 80, 170), 'fruiting': (120, 65, 145),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'January-March'
    },
    'Banana': {
        'vegetative': (125, 65, 145), 'flowering': (145, 75, 165), 'fruiting': (120, 60, 140),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'January-March'
    },
    'Pineapple': {
        'vegetative': (95, 45, 105), 'flowering': (110, 55, 120), 'fruiting': (90, 40, 100),
        'ph_min': 5.0, 'ph_max': 6.0, 'season': 'March-May'
    },
    'Mango': {
        'vegetative': (80, 50, 90), 'flowering': (95, 60, 105), 'fruiting': (75, 45, 85),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    'Orange': {
        'vegetative': (105, 55, 95), 'flowering': (120, 65, 110), 'fruiting': (100, 50, 90),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    
    # TREE CROPS
    'Cashew': {
        'vegetative': (70, 40, 80), 'flowering': (85, 50, 95), 'nut_development': (65, 35, 75),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'February-April'
    },
    'Oil Palm': {
        'vegetative': (100, 60, 120), 'flowering': (120, 70, 140), 'fruit_development': (95, 55, 115),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'January-March'
    },
    'Cocoa': {
        'vegetative': (85, 55, 95), 'flowering': (100, 65, 110), 'pod_development': (80, 50, 90),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    'Coffee': {
        'vegetative': (90, 50, 85), 'flowering': (105, 60, 100), 'berry_development': (85, 45, 80),
        'ph_min': 5.5, 'ph_max': 6.5, 'season': 'March-May'
    },
    
    # INDUSTRIAL CROPS
    'Cotton': {
        'vegetative': (100, 50, 90), 'flowering': (120, 60, 110), 'boll_development': (90, 45, 85),
        'ph_min': 6.0, 'ph_max': 7.5, 'season': 'June-July'
    },
    'Sesame': {
        'vegetative': (55, 45, 65), 'flowering': (70, 55, 80), 'capsule_filling': (50, 40, 60),
        'ph_min': 6.0, 'ph_max': 7.5, 'season': 'June-July'
    },
}

# ============================================
# 2. REGIONAL ADJUSTMENT FACTORS
# Format: {region_name: (climate_zone, n_factor, p_factor, k_factor, ph_adjustment)}
# ============================================
REGIONAL_FACTORS = {
    # TROPICAL COASTAL - High rainfall, acidic soils
    'Lagos': ('Tropical Coastal', 1.05, 1.02, 1.08, -0.1),
    'Port Harcourt': ('Tropical Coastal', 1.08, 1.05, 1.10, -0.1),
    'Warri': ('Tropical Coastal', 1.07, 1.04, 1.09, -0.1),
    'Yenagoa': ('Tropical Coastal', 1.08, 1.05, 1.10, -0.15),
    'Uyo': ('Tropical Coastal', 1.07, 1.04, 1.09, -0.1),
    'Calabar': ('Tropical Coastal', 1.08, 1.05, 1.10, -0.1),
    
    # TROPICAL RAINFOREST
    'Ile-Ife': ('Tropical Rainforest', 1.03, 1.02, 1.05, -0.05),
    'Akure': ('Tropical Rainforest', 1.04, 1.03, 1.06, -0.05),
    'Ado-Ekiti': ('Tropical Rainforest', 1.02, 1.01, 1.04, -0.05),
    'Enugu': ('Tropical Rainforest', 1.03, 1.02, 1.05, -0.05),
    'Owerri': ('Tropical Rainforest', 1.04, 1.03, 1.06, -0.05),
    'Umuahia': ('Tropical Rainforest', 1.03, 1.02, 1.05, -0.05),
    'Awka': ('Tropical Rainforest', 1.03, 1.02, 1.05, -0.05),
    'Abakaliki': ('Tropical Rainforest', 1.02, 1.01, 1.04, -0.05),
    'Benin City': ('Tropical Rainforest', 1.04, 1.03, 1.06, -0.05),
    
    # TROPICAL SAVANNA
    'Ibadan': ('Tropical Savanna', 1.0, 1.0, 1.0, 0.0),
    'Osogbo': ('Tropical Savanna', 0.98, 0.98, 0.98, 0.0),
    'Ilorin': ('Tropical Savanna', 0.97, 0.97, 0.97, 0.05),
    'Lokoja': ('Tropical Savanna', 0.98, 0.98, 0.98, 0.0),
    'Makurdi': ('Tropical Savanna', 0.99, 0.99, 0.99, 0.0),
    'Yola': ('Tropical Savanna', 0.97, 0.97, 0.97, 0.05),
    'Jalingo': ('Tropical Savanna', 0.98, 0.98, 0.98, 0.0),
    
    # SAVANNA
    'Abuja': ('Savanna', 0.95, 0.95, 0.95, 0.1),
    'Minna': ('Savanna', 0.93, 0.93, 0.93, 0.1),
    'Kaduna': ('Savanna', 0.96, 0.96, 0.96, 0.05),
    'Zaria': ('Savanna', 0.95, 0.95, 0.95, 0.05),
    'Gombe': ('Savanna', 0.94, 0.94, 0.94, 0.1),
    
    # SAHEL SAVANNA
    'Kano': ('Sahel Savanna', 0.90, 0.88, 0.90, 0.3),
    'Sokoto': ('Sahel Savanna', 0.88, 0.85, 0.88, 0.4),
    'Katsina': ('Sahel Savanna', 0.89, 0.87, 0.89, 0.35),
    'Gusau': ('Sahel Savanna', 0.89, 0.87, 0.89, 0.3),
    'Birnin Kebbi': ('Sahel Savanna', 0.88, 0.86, 0.88, 0.35),
    'Bauchi': ('Sahel Savanna', 0.90, 0.88, 0.90, 0.3),
    
    # SAHEL
    'Maiduguri': ('Sahel', 0.85, 0.82, 0.85, 0.5),
    'Damaturu': ('Sahel', 0.84, 0.81, 0.84, 0.5),
    
    # HIGHLAND SAVANNA
    'Jos': ('Highland Savanna', 0.95, 0.93, 0.95, 0.15),
}

# ============================================
# 3. GENERATE ALL COMBINATIONS
# ============================================
def generate_all_requirements():
    """Generate nutrient requirements for every crop in every region"""
    records = []
    
    for crop_name, base_req in CROP_BASE_REQUIREMENTS.items():
        for region_name, (climate, n_fact, p_fact, k_fact, ph_adj) in REGIONAL_FACTORS.items():
            # Get state from region name (simplified mapping)
            state = region_name  # In production, you'd look this up from regional_defaults
            
            # Apply regional adjustments for each growth stage
            for stage, (n_base, p_base, k_base) in [
                ('vegetative', base_req.get('vegetative', (0,0,0))),
                ('flowering', base_req.get('flowering', base_req.get('fruiting', (0,0,0)))),
                ('fruiting', base_req.get('fruiting', base_req.get('maturity', (0,0,0)))),
                ('maturity', base_req.get('maturity', (0,0,0))),
                ('tuber_bulking', base_req.get('tuber_bulking', (0,0,0))),
                ('tuber_initiation', base_req.get('tuber_initiation', (0,0,0))),
                ('corm_development', base_req.get('corm_development', (0,0,0))),
                ('head_formation', base_req.get('head_formation', (0,0,0))),
                ('bulb_formation', base_req.get('bulb_formation', (0,0,0))),
                ('root_development', base_req.get('root_development', (0,0,0))),
                ('nut_development', base_req.get('nut_development', (0,0,0))),
                ('fruit_development', base_req.get('fruit_development', (0,0,0))),
                ('pod_development', base_req.get('pod_development', (0,0,0))),
                ('berry_development', base_req.get('berry_development', (0,0,0))),
                ('boll_development', base_req.get('boll_development', (0,0,0))),
                ('capsule_filling', base_req.get('capsule_filling', (0,0,0))),
            ]:
                if n_base == 0 and p_base == 0 and k_base == 0:
                    continue  # Skip if stage doesn't exist for this crop
                
                # Apply regional factors
                n_req = round(n_base * n_fact, 0)
                p_req = round(p_base * p_fact, 0)
                k_req = round(k_base * k_fact, 0)
                
                # Adjust pH range
                ph_min = round(base_req['ph_min'] + ph_adj, 1)
                ph_max = round(base_req['ph_max'] + ph_adj, 1)
                
                # Ensure pH stays in reasonable range
                ph_min = max(4.5, min(8.5, ph_min))
                ph_max = max(4.5, min(8.5, ph_max))
                
                records.append((
                    crop_name, region_name, state,
                    int(n_req), int(p_req), int(k_req),
                    ph_min, ph_max, stage, base_req['season']
                ))
    
    return records

# ============================================
# 4. INSERT INTO DATABASE
# ============================================
def populate_database():
    print("🌱 Starting database population...")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Clear existing data
    print("️ Clearing existing crop_nutrient_requirements...")
    conn.execute('DELETE FROM crop_nutrient_requirements')
    
    # Generate all combinations
    print("🧮 Generating all crop × region combinations...")
    records = generate_all_requirements()
    
    print(f" Total records to insert: {len(records)}")
    print(f"   - {len(CROP_BASE_REQUIREMENTS)} crops")
    print(f"   - {len(REGIONAL_FACTORS)} regions")
    print(f"   - Average stages per crop: {len(records) / len(CROP_BASE_REQUIREMENTS) / len(REGIONAL_FACTORS):.1f}")
    print()
    
    # Insert records
    print(" Inserting records into database...")
    inserted = 0
    errors = 0
    
    for record in records:
        try:
            conn.execute('''INSERT INTO crop_nutrient_requirements 
                (crop_type, region, state, n_requirement, p_requirement, k_requirement,
                 optimal_ph_min, optimal_ph_max, growth_stage, planting_season)
                VALUES (?,?,?,?,?,?,?,?,?,?)''', record)
            inserted += 1
        except sqlite3.IntegrityError as e:
            errors += 1
            if errors <= 5:  # Only show first few errors
                print(f"⚠️ Skip: {record[0]} in {record[1]} - {e}")
    
    conn.commit()
    conn.close()
    
    print()
    print("=" * 60)
    print("✅ POPULATION COMPLETE!")
    print(f"   Successfully inserted: {inserted} records")
    print(f"   Skipped (duplicates): {errors} records")
    print()
    
    # Verify
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute('SELECT COUNT(*) FROM crop_nutrient_requirements').fetchone()[0]
    unique_crops = conn.execute('SELECT COUNT(DISTINCT crop_type) FROM crop_nutrient_requirements').fetchone()[0]
    unique_regions = conn.execute('SELECT COUNT(DISTINCT region) FROM crop_nutrient_requirements').fetchone()[0]
    conn.close()
    
    print("📊 VERIFICATION:")
    print(f"   Total records in DB: {total}")
    print(f"   Unique crops: {unique_crops}")
    print(f"   Unique regions: {unique_regions}")
    print()
    print("🎉 All crops now have requirements for ALL regions!")

if __name__ == '__main__':
    populate_database()