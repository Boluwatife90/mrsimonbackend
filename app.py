# -*- coding: utf-8 -*-
import os
import sqlite3
import traceback
import numpy as np
import pandas as pd
import shutil
import requests
from functools import wraps
from datetime import timedelta, datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
from werkzeug.security import generate_password_hash, check_password_hash

# --- ML Model Loading ---
try:
    from phase3_predict_corrected_shimmed import predict_with_experts
    ML_LOADED = True
    print("✅ Stacked Ensemble Model loaded successfully.")
except ImportError:
    print("⚠️ Warning: 'phase3_predict_corrected_shimmed.py' not found.")
    ML_LOADED = False

def predict_with_experts(payload, need_threshold=0.45):
    soil_moisture = payload.get('soil_moisture', 35)
    n_val = payload.get('N', 25)
    p_val = payload.get('P', 18)
    k_val = payload.get('K', 180)
    ndvi = payload.get('ndvi_proxy', 0.65)

    nutrient_stress = (n_val < 30 or p_val < 15 or k_val < 100)
    water_stress = (soil_moisture < 25 or soil_moisture > 50)
    crop_stress = (ndvi < 0.5)
    need = (nutrient_stress and not water_stress) or crop_stress

    n_deficit = max(0, 30 - n_val)
    p_deficit = max(0, 15 - p_val)
    k_deficit = max(0, 100 - k_val)
    base_rate = (n_deficit * 2.5) + (p_deficit * 3.0) + (k_deficit * 1.5)
    base_rate = min(base_rate, 200)

    stress_score = sum([nutrient_stress, water_stress, crop_stress])
    base_proba = 0.50 + (stress_score * 0.15)
    base_proba = min(base_proba, 0.95)

    if soil_moisture < 20 or ndvi < 0.3:
        timing = "24_48h"
    elif soil_moisture < 30 or ndvi < 0.5:
        timing = "48_72h"
    else:
        timing = "72_96h"

    return {
        "need_label": 1 if need else 0,
        "need_proba": round(base_proba, 3),
        "rate_pred": round(base_rate, 1),
        "timing": timing,
        "expert": {
            "base": {
                "ts_pred_soil_moisture": round(soil_moisture * 0.95, 1),
                "base_rate_raw": round(base_rate * 0.98, 1),
                "base_need_proba": round(base_proba * 0.97, 3)
            }
        }
    }

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pfdss_users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Users Table
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT DEFAULT 'user'
    )''')
    try:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        conn.commit()
    except Exception:
        pass

    # 2. Feedback & Sensor Tables
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        comments TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT NOT NULL,
        soil_moisture REAL, soil_temp_c REAL, soil_ph REAL,
        nutrient_ec_dS_m REAL, npk_n_mgkg REAL, npk_p_mgkg REAL, npk_k_mgkg REAL,
        air_temp_c REAL, humidity_pct REAL, rainfall_forecast_mm REAL,
        crop_age_days REAL, plant_vi_proxy REAL, crop_type TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # 3. System Settings Table
    conn.execute('''CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )''')
    defaults = [
        ('need_threshold', '0.45'),
        ('base_rate', '100.0'),
        ('allowed_crops', 'maize,rice,tomato,cassava,pepper')
    ]
    for key, value in defaults:
        conn.execute('INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)', (key, value))

    # 4. Prediction Logs Table
    conn.execute('''CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        crop_type TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # 5. Crops Management Table
    conn.execute('''CREATE TABLE IF NOT EXISTS crops (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT UNIQUE NOT NULL,
        scientific_name TEXT, 
        n_req REAL, p_req REAL, k_req REAL
    )''')
    default_crops = [
        ('Maize', 'Zea mays', 120, 60, 80),
        ('Rice', 'Oryza sativa', 100, 50, 70),
        ('Tomato', 'Solanum lycopersicum', 150, 80, 120),
        ('Cassava', 'Manihot esculenta', 80, 40, 100),
        ('Pepper', 'Capsicum spp.', 110, 60, 90),
    
        ('Yam', 'Dioscorea spp.', 90, 45, 110),
        ('Sorghum', 'Sorghum bicolor', 80, 40, 60),
        ('Millet', 'Pennisetum glaucum', 60, 30, 50),
        ('Groundnut', 'Arachis hypogaea', 50, 60, 90),
        ('Cowpea', 'Vigna unguiculata', 40, 50, 70),
        ('Soybean', 'Glycine max', 60, 70, 80),
        ('Sweet Potato', 'Ipomoea batatas', 70, 50, 100),
        ('Cocoyam', 'Colocasia esculenta', 85, 45, 105),
        ('Plantain', 'Musa paradisiaca', 130, 70, 150),
        ('Cotton', 'Gossypium hirsutum', 100, 50, 90),
        ('Sesame', 'Sesamum indicum', 55, 45, 65),
        ('Melon', 'Citrullus lanatus', 75, 55, 85),
        ('Okra', 'Abelmoschus esculentus', 95, 55, 85),
        ('Cabbage', 'Brassica oleracea', 140, 70, 110),
        ('Onion', 'Allium cepa', 110, 65, 95),
        ('Irish Potato', 'Solanum tuberosum', 120, 75, 130),
        ('Wheat', 'Triticum aestivum', 90, 50, 70),
        ('Cashew', 'Anacardium occidentale', 70, 40, 80),
        ('Oil Palm', 'Elaeis guineensis', 100, 60, 120),
        ('Cocoa', 'Theobroma cacao', 85, 55, 95),
        ('Coffee', 'Coffea arabica', 90, 50, 85),
        ('Banana', 'Musa sapientum', 125, 65, 145),
        ('Carrot', 'Daucus carota', 85, 60, 95),
        ('Cucumber', 'Cucumis sativus', 90, 55, 85),
        ('Lettuce', 'Lactuca sativa', 100, 50, 80),
        ('Watermelon', 'Citrullus lanatus', 80, 60, 90),
        ('Pineapple', 'Ananas comosus', 95, 45, 105),
        ('Mango', 'Mangifera indica', 80, 50, 90),
        ('Orange', 'Citrus sinensis', 105, 55, 95),
        ('Cashew', 'Anacardium occidentale', 75, 45, 85)
    ]
    
    for c in default_crops:
        conn.execute('INSERT OR IGNORE INTO crops (name, scientific_name, n_req, p_req, k_req) VALUES (?,?,?,?,?)', c)

    # 6. Fertilizers Management Table
    conn.execute('''CREATE TABLE IF NOT EXISTS fertilizers (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT UNIQUE NOT NULL,
        n_pct REAL, p_pct REAL, k_pct REAL, 
        price_per_kg REAL
    )''')
    default_ferts = [
        ('Urea', 46, 0, 0, 950),
        ('DAP', 18, 46, 0, 1120),
        ('MOP', 0, 0, 60, 870),
        ('NPK 15-15-15', 15, 15, 15, 1050)
    ]
    for f in default_ferts:
        conn.execute('INSERT OR IGNORE INTO fertilizers (name, n_pct, p_pct, k_pct, price_per_kg) VALUES (?,?,?,?,?)', f)

    # 7. Regional Soil Database Table
    conn.execute('''CREATE TABLE IF NOT EXISTS regional_defaults (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region_name TEXT UNIQUE NOT NULL,
        state TEXT,
        default_ph REAL,
        default_moisture REAL,
        typical_n REAL,
        typical_p REAL,
        typical_k REAL,
        climate_zone TEXT,
        rainfall_zone TEXT
    )''')

    default_regions = [
        ('Lagos', 'Lagos', 6.0, 35, 25, 18, 180, 'Tropical Coastal', 'High Rainfall (1800-2200mm)'),
        ('Ibadan', 'Oyo', 6.2, 32, 22, 16, 170, 'Tropical Savanna', 'Moderate Rainfall (1200-1500mm)'),
        ('Ile-Ife', 'Osun', 6.1, 33, 23, 17, 175, 'Tropical Rainforest', 'High Rainfall (1400-1600mm)'),
        ('Akure', 'Ondo', 5.9, 34, 24, 18, 185, 'Tropical Rainforest', 'High Rainfall (1500-1700mm)'),
        ('Ado-Ekiti', 'Ekiti', 6.0, 31, 21, 15, 165, 'Tropical Rainforest', 'Moderate Rainfall (1300-1500mm)'),
        ('Osogbo', 'Osun', 6.2, 30, 20, 14, 160, 'Tropical Savanna', 'Moderate Rainfall (1200-1400mm)'),
        ('Enugu', 'Enugu', 6.3, 33, 23, 16, 165, 'Tropical Rainforest', 'High Rainfall (1500-1700mm)'),
        ('Port Harcourt', 'Rivers', 5.8, 38, 28, 20, 190, 'Tropical Coastal', 'Very High Rainfall (2000-2400mm)'),
        ('Owerri', 'Imo', 6.1, 35, 25, 18, 175, 'Tropical Rainforest', 'High Rainfall (1800-2000mm)'),
        ('Umuahia', 'Abia', 6.0, 34, 24, 17, 170, 'Tropical Rainforest', 'High Rainfall (1700-1900mm)'),
        ('Awka', 'Anambra', 6.2, 32, 22, 16, 165, 'Tropical Rainforest', 'High Rainfall (1600-1800mm)'),
        ('Abakaliki', 'Ebonyi', 6.1, 31, 21, 15, 160, 'Tropical Rainforest', 'Moderate Rainfall (1500-1700mm)'),
        ('Benin City', 'Edo', 6.1, 34, 24, 17, 175, 'Tropical Rainforest', 'High Rainfall (1800-2100mm)'),
        ('Warri', 'Delta', 5.9, 36, 26, 19, 185, 'Tropical Coastal', 'Very High Rainfall (2000-2400mm)'),
        ('Yenagoa', 'Bayelsa', 5.7, 37, 27, 20, 190, 'Tropical Coastal', 'Very High Rainfall (2200-2600mm)'),
        ('Uyo', 'Akwa Ibom', 5.8, 36, 26, 19, 185, 'Tropical Coastal', 'Very High Rainfall (2000-2400mm)'),
        ('Calabar', 'Cross River', 5.9, 37, 27, 20, 190, 'Tropical Coastal', 'Very High Rainfall (2500-3000mm)'),
        ('Abuja', 'FCT', 6.5, 30, 20, 15, 150, 'Savanna', 'Moderate Rainfall (1100-1300mm)'),
        ('Ilorin', 'Kwara', 6.4, 29, 19, 14, 145, 'Tropical Savanna', 'Moderate Rainfall (1100-1300mm)'),
        ('Lokoja', 'Kogi', 6.3, 30, 20, 15, 150, 'Tropical Savanna', 'Moderate Rainfall (1200-1400mm)'),
        ('Makurdi', 'Benue', 6.4, 31, 21, 16, 155, 'Tropical Savanna', 'Moderate Rainfall (1100-1300mm)'),
        ('Minna', 'Niger', 6.5, 28, 18, 13, 140, 'Savanna', 'Moderate Rainfall (1000-1200mm)'),
        ('Jos', 'Plateau', 6.6, 27, 17, 12, 135, 'Highland Savanna', 'Moderate Rainfall (1300-1500mm)'),
        ('Kano', 'Kano', 7.0, 25, 18, 12, 140, 'Sahel Savanna', 'Low Rainfall (600-800mm)'),
        ('Kaduna', 'Kaduna', 6.8, 28, 19, 14, 145, 'Savanna', 'Moderate Rainfall (900-1100mm)'),
        ('Sokoto', 'Sokoto', 7.2, 24, 16, 11, 130, 'Sahel Savanna', 'Low Rainfall (500-700mm)'),
        ('Zaria', 'Kaduna', 6.9, 27, 18, 13, 140, 'Savanna', 'Moderate Rainfall (900-1100mm)'),
        ('Katsina', 'Katsina', 7.1, 25, 17, 12, 135, 'Sahel Savanna', 'Low Rainfall (600-800mm)'),
        ('Gusau', 'Zamfara', 7.0, 26, 17, 12, 135, 'Sahel Savanna', 'Low Rainfall (700-900mm)'),
        ('Birnin Kebbi', 'Kebbi', 7.1, 25, 16, 11, 130, 'Sahel Savanna', 'Low Rainfall (600-800mm)'),
        ('Maiduguri', 'Borno', 7.2, 22, 15, 10, 130, 'Sahel', 'Very Low Rainfall (400-600mm)'),
        ('Yola', 'Adamawa', 6.9, 26, 17, 12, 135, 'Savanna', 'Moderate Rainfall (800-1000mm)'),
        ('Bauchi', 'Bauchi', 7.0, 25, 16, 11, 130, 'Sahel Savanna', 'Low Rainfall (700-900mm)'),
        ('Gombe', 'Gombe', 6.9, 26, 17, 12, 135, 'Sahel Savanna', 'Low Rainfall (700-900mm)'),
        ('Damaturu', 'Yobe', 7.3, 23, 15, 10, 125, 'Sahel', 'Very Low Rainfall (400-600mm)'),
        ('Jalingo', 'Taraba', 6.7, 28, 18, 13, 140, 'Savanna', 'Moderate Rainfall (1000-1200mm)')
    ]

    for region in default_regions:
        conn.execute('''INSERT OR IGNORE INTO regional_defaults 
            (region_name, state, default_ph, default_moisture, typical_n, typical_p, typical_k, climate_zone, rainfall_zone) 
            VALUES (?,?,?,?,?,?,?,?,?)''', region)

    # 8. Seasonal Calendar Table
    conn.execute('''CREATE TABLE IF NOT EXISTS seasonal_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_type TEXT NOT NULL,
    region TEXT NOT NULL,
    state TEXT,
    planting_start TEXT,
    planting_end TEXT,
    harvest_start TEXT,
    harvest_end TEXT,
    rainy_season_start TEXT,
    rainy_season_end TEXT,
    growing_days INTEGER,
    UNIQUE(crop_type, region, state)
                
)''')
    default_seasons = [
        ('Maize', 'Lagos', 'Lagos', 'March', 'April', 'July', 'August', 'March', 'November', 90),
        ('Maize', 'Abuja', 'FCT', 'April', 'May', 'August', 'September', 'April', 'October', 90),
        ('Maize', 'Kano', 'Kano', 'May', 'June', 'September', 'October', 'June', 'September', 90),
        ('Maize', 'Ibadan', 'Oyo', 'March', 'April', 'July', 'August', 'March', 'November', 90),
        ('Maize', 'Port Harcourt', 'Rivers', 'March', 'April', 'July', 'August', 'March', 'December', 90),
        ('Maize', 'Enugu', 'Enugu', 'March', 'April', 'July', 'August', 'March', 'November', 90),
        ('Maize', 'Kaduna', 'Kaduna', 'May', 'June', 'September', 'October', 'May', 'October', 90),
        ('Maize', 'Benin City', 'Edo', 'March', 'April', 'July', 'August', 'March', 'November', 90),
        ('Maize', 'Maiduguri', 'Borno', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Maize', 'Ilorin', 'Kwara', 'April', 'May', 'August', 'September', 'April', 'October', 90),
        ('Rice', 'Lagos', 'Lagos', 'April', 'May', 'September', 'October', 'March', 'November', 120),
        ('Rice', 'Abuja', 'FCT', 'May', 'June', 'October', 'November', 'April', 'October', 120),
        ('Rice', 'Kano', 'Kano', 'June', 'July', 'November', 'December', 'June', 'September', 120),
        ('Rice', 'Port Harcourt', 'Rivers', 'April', 'May', 'September', 'October', 'March', 'December', 120),
        ('Rice', 'Enugu', 'Enugu', 'April', 'May', 'September', 'October', 'March', 'November', 120),
        ('Rice', 'Benin City', 'Edo', 'April', 'May', 'September', 'October', 'March', 'November', 120),
        ('Rice', 'Ibadan', 'Oyo', 'April', 'May', 'September', 'October', 'March', 'November', 120),
        ('Rice', 'Kaduna', 'Kaduna', 'June', 'July', 'November', 'December', 'May', 'October', 120),
        ('Rice', 'Makurdi', 'Benue', 'May', 'June', 'October', 'November', 'April', 'October', 120),
        ('Rice', 'Jos', 'Plateau', 'May', 'June', 'October', 'November', 'April', 'October', 120),
        ('Cassava', 'Lagos', 'Lagos', 'January', 'March', 'January', 'December', 'March', 'November', 365),
        ('Cassava', 'Abuja', 'FCT', 'February', 'April', 'February', 'December', 'April', 'October', 365),
        ('Cassava', 'Ibadan', 'Oyo', 'January', 'March', 'January', 'December', 'March', 'November', 365),
        ('Cassava', 'Port Harcourt', 'Rivers', 'January', 'March', 'January', 'December', 'March', 'December', 365),
        ('Cassava', 'Enugu', 'Enugu', 'February', 'April', 'February', 'December', 'March', 'November', 365),
        ('Cassava', 'Benin City', 'Edo', 'January', 'March', 'January', 'December', 'March', 'November', 365),
        ('Cassava', 'Kano', 'Kano', 'March', 'May', 'March', 'December', 'June', 'September', 365),
        ('Cassava', 'Kaduna', 'Kaduna', 'March', 'May', 'March', 'December', 'May', 'October', 365),
        ('Cassava', 'Calabar', 'Cross River', 'January', 'March', 'January', 'December', 'March', 'December', 365),
        ('Cassava', 'Owerri', 'Imo', 'February', 'April', 'February', 'December', 'March', 'November', 365),
        ('Yam', 'Abuja', 'FCT', 'February', 'April', 'August', 'October', 'April', 'October', 180),
        ('Yam', 'Ibadan', 'Oyo', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        ('Yam', 'Enugu', 'Enugu', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        ('Yam', 'Makurdi', 'Benue', 'February', 'April', 'August', 'October', 'April', 'October', 180),
        ('Yam', 'Ilorin', 'Kwara', 'March', 'May', 'September', 'November', 'April', 'October', 180),
        ('Yam', 'Lokoja', 'Kogi', 'February', 'April', 'August', 'October', 'April', 'October', 180),
        ('Yam', 'Minna', 'Niger', 'March', 'May', 'September', 'November', 'April', 'October', 180),
        ('Yam', 'Jos', 'Plateau', 'March', 'May', 'September', 'November', 'April', 'October', 180),
        ('Yam', 'Kaduna', 'Kaduna', 'March', 'May', 'September', 'November', 'May', 'October', 180),
        ('Yam', 'Owerri', 'Imo', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        ('Tomato', 'Kano', 'Kano', 'September', 'October', 'January', 'February', 'June', 'September', 120),
        ('Tomato', 'Kaduna', 'Kaduna', 'September', 'October', 'January', 'February', 'May', 'October', 120),
        ('Tomato', 'Abuja', 'FCT', 'October', 'November', 'February', 'March', 'April', 'October', 120),
        ('Tomato', 'Jos', 'Plateau', 'September', 'October', 'January', 'February', 'April', 'October', 120),
        ('Tomato', 'Sokoto', 'Sokoto', 'October', 'November', 'February', 'March', 'July', 'September', 120),
        ('Tomato', 'Zaria', 'Kaduna', 'September', 'October', 'January', 'February', 'May', 'October', 120),
        ('Tomato', 'Ibadan', 'Oyo', 'October', 'November', 'February', 'March', 'March', 'November', 120),
        ('Tomato', 'Ilorin', 'Kwara', 'October', 'November', 'February', 'March', 'April', 'October', 120),
        ('Tomato', 'Maiduguri', 'Borno', 'November', 'December', 'March', 'April', 'July', 'September', 120),
        ('Tomato', 'Bauchi', 'Bauchi', 'October', 'November', 'February', 'March', 'July', 'September', 120),
        ('Pepper', 'Lagos', 'Lagos', 'September', 'October', 'January', 'February', 'March', 'November', 120),
        ('Pepper', 'Kano', 'Kano', 'August', 'September', 'December', 'January', 'June', 'September', 120),
        ('Pepper', 'Abuja', 'FCT', 'September', 'October', 'January', 'February', 'April', 'October', 120),
        ('Pepper', 'Ibadan', 'Oyo', 'September', 'October', 'January', 'February', 'March', 'November', 120),
        ('Pepper', 'Port Harcourt', 'Rivers', 'September', 'October', 'January', 'February', 'March', 'December', 120),
        ('Pepper', 'Enugu', 'Enugu', 'September', 'October', 'January', 'February', 'March', 'November', 120),
        ('Pepper', 'Kaduna', 'Kaduna', 'August', 'September', 'December', 'January', 'May', 'October', 120),
        ('Pepper', 'Benin City', 'Edo', 'September', 'October', 'January', 'February', 'March', 'November', 120),
        ('Pepper', 'Ilorin', 'Kwara', 'September', 'October', 'January', 'February', 'April', 'October', 120),
        ('Pepper', 'Jos', 'Plateau', 'August', 'September', 'December', 'January', 'April', 'October', 120),
        ('Sorghum', 'Kano', 'Kano', 'May', 'June', 'September', 'October', 'June', 'September', 100),
        ('Sorghum', 'Kaduna', 'Kaduna', 'May', 'June', 'September', 'October', 'May', 'October', 100),
        ('Sorghum', 'Sokoto', 'Sokoto', 'June', 'July', 'October', 'November', 'July', 'September', 100),
        ('Sorghum', 'Maiduguri', 'Borno', 'June', 'July', 'October', 'November', 'July', 'September', 100),
        ('Sorghum', 'Zaria', 'Kaduna', 'May', 'June', 'September', 'October', 'May', 'October', 100),
        ('Sorghum', 'Katsina', 'Katsina', 'May', 'June', 'September', 'October', 'June', 'September', 100),
        ('Sorghum', 'Bauchi', 'Bauchi', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Sorghum', 'Gombe', 'Gombe', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Sorghum', 'Yola', 'Adamawa', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Sorghum', 'Abuja', 'FCT', 'May', 'June', 'September', 'October', 'April', 'October', 100),
        ('Millet', 'Kano', 'Kano', 'June', 'July', 'October', 'November', 'June', 'September', 90),
        ('Millet', 'Sokoto', 'Sokoto', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Maiduguri', 'Borno', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Katsina', 'Katsina', 'June', 'July', 'October', 'November', 'June', 'September', 90),
        ('Millet', 'Zaria', 'Kaduna', 'June', 'July', 'October', 'November', 'May', 'October', 90),
        ('Millet', 'Gusau', 'Zamfara', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Bauchi', 'Bauchi', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Damaturu', 'Yobe', 'July', 'August', 'November', 'December', 'July', 'September', 90),
        ('Millet', 'Kaduna', 'Kaduna', 'June', 'July', 'October', 'November', 'May', 'October', 90),
        ('Millet', 'Abuja', 'FCT', 'June', 'July', 'October', 'November', 'April', 'October', 90),
    
        ('Yam', 'Lagos', 'Lagos', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        ('Yam', 'Abuja', 'FCT', 'February', 'April', 'August', 'October', 'April', 'October', 180),
        ('Yam', 'Kano', 'Kano', 'March', 'May', 'September', 'November', 'June', 'September', 180),
        ('Yam', 'Ibadan', 'Oyo', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        ('Yam', 'Enugu', 'Enugu', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        ('Yam', 'Makurdi', 'Benue', 'February', 'April', 'August', 'October', 'April', 'October', 180),
        ('Yam', 'Ilorin', 'Kwara', 'March', 'May', 'September', 'November', 'April', 'October', 180),
        ('Yam', 'Owerri', 'Imo', 'February', 'April', 'August', 'October', 'March', 'November', 180),
        
        # SORGHUM
        ('Sorghum', 'Kano', 'Kano', 'May', 'June', 'September', 'October', 'June', 'September', 100),
        ('Sorghum', 'Kaduna', 'Kaduna', 'May', 'June', 'September', 'October', 'May', 'October', 100),
        ('Sorghum', 'Sokoto', 'Sokoto', 'June', 'July', 'October', 'November', 'July', 'September', 100),
        ('Sorghum', 'Maiduguri', 'Borno', 'June', 'July', 'October', 'November', 'July', 'September', 100),
        ('Sorghum', 'Zaria', 'Kaduna', 'May', 'June', 'September', 'October', 'May', 'October', 100),
        ('Sorghum', 'Katsina', 'Katsina', 'May', 'June', 'September', 'October', 'June', 'September', 100),
        ('Sorghum', 'Bauchi', 'Bauchi', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Sorghum', 'Gombe', 'Gombe', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Sorghum', 'Yola', 'Adamawa', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Sorghum', 'Abuja', 'FCT', 'May', 'June', 'September', 'October', 'April', 'October', 100),
        
        # MILLET
        ('Millet', 'Kano', 'Kano', 'June', 'July', 'October', 'November', 'June', 'September', 90),
        ('Millet', 'Sokoto', 'Sokoto', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Maiduguri', 'Borno', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Katsina', 'Katsina', 'June', 'July', 'October', 'November', 'June', 'September', 90),
        ('Millet', 'Zaria', 'Kaduna', 'June', 'July', 'October', 'November', 'May', 'October', 90),
        ('Millet', 'Gusau', 'Zamfara', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Bauchi', 'Bauchi', 'June', 'July', 'October', 'November', 'July', 'September', 90),
        ('Millet', 'Damaturu', 'Yobe', 'July', 'August', 'November', 'December', 'July', 'September', 90),
        ('Millet', 'Kaduna', 'Kaduna', 'June', 'July', 'October', 'November', 'May', 'October', 90),
        ('Millet', 'Abuja', 'FCT', 'June', 'July', 'October', 'November', 'April', 'October', 90),
        
        # GROUNDNUT
        ('Groundnut', 'Kano', 'Kano', 'June', 'July', 'October', 'November', 'June', 'September', 110),
        ('Groundnut', 'Kaduna', 'Kaduna', 'June', 'July', 'October', 'November', 'May', 'October', 110),
        ('Groundnut', 'Sokoto', 'Sokoto', 'June', 'July', 'October', 'November', 'July', 'September', 110),
        ('Groundnut', 'Katsina', 'Katsina', 'June', 'July', 'October', 'November', 'June', 'September', 110),
        ('Groundnut', 'Bauchi', 'Bauchi', 'June', 'July', 'October', 'November', 'July', 'September', 110),
        ('Groundnut', 'Gombe', 'Gombe', 'June', 'July', 'October', 'November', 'July', 'September', 110),
        ('Groundnut', 'Maiduguri', 'Borno', 'June', 'July', 'October', 'November', 'July', 'September', 110),
        ('Groundnut', 'Zaria', 'Kaduna', 'June', 'July', 'October', 'November', 'May', 'October', 110),
        ('Groundnut', 'Abuja', 'FCT', 'June', 'July', 'October', 'November', 'April', 'October', 110),
        
        # COWPEA (BEANS)
        ('Cowpea', 'Kano', 'Kano', 'June', 'July', 'September', 'October', 'June', 'September', 75),
        ('Cowpea', 'Kaduna', 'Kaduna', 'June', 'July', 'September', 'October', 'May', 'October', 75),
        ('Cowpea', 'Sokoto', 'Sokoto', 'June', 'July', 'September', 'October', 'July', 'September', 75),
        ('Cowpea', 'Katsina', 'Katsina', 'June', 'July', 'September', 'October', 'June', 'September', 75),
        ('Cowpea', 'Bauchi', 'Bauchi', 'June', 'July', 'September', 'October', 'July', 'September', 75),
        ('Cowpea', 'Gombe', 'Gombe', 'June', 'July', 'September', 'October', 'July', 'September', 75),
        ('Cowpea', 'Maiduguri', 'Borno', 'June', 'July', 'September', 'October', 'July', 'September', 75),
        ('Cowpea', 'Abuja', 'FCT', 'June', 'July', 'September', 'October', 'April', 'October', 75),
        ('Cowpea', 'Ibadan', 'Oyo', 'April', 'May', 'July', 'August', 'March', 'November', 75),
        
        # SOYBEAN
        ('Soybean', 'Abuja', 'FCT', 'May', 'June', 'September', 'October', 'April', 'October', 100),
        ('Soybean', 'Kaduna', 'Kaduna', 'May', 'June', 'September', 'October', 'May', 'October', 100),
        ('Soybean', 'Kano', 'Kano', 'June', 'July', 'September', 'October', 'June', 'September', 100),
        ('Soybean', 'Bauchi', 'Bauchi', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Soybean', 'Gombe', 'Gombe', 'May', 'June', 'September', 'October', 'July', 'September', 100),
        ('Soybean', 'Ibadan', 'Oyo', 'April', 'May', 'August', 'September', 'March', 'November', 100),
        ('Soybean', 'Ilorin', 'Kwara', 'May', 'June', 'September', 'October', 'April', 'October', 100),
        
        # SWEET POTATO
        ('Sweet Potato', 'Lagos', 'Lagos', 'March', 'May', 'August', 'October', 'March', 'November', 120),
        ('Sweet Potato', 'Abuja', 'FCT', 'March', 'May', 'August', 'October', 'April', 'October', 120),
        ('Sweet Potato', 'Ibadan', 'Oyo', 'March', 'May', 'August', 'October', 'March', 'November', 120),
        ('Sweet Potato', 'Enugu', 'Enugu', 'March', 'May', 'August', 'October', 'March', 'November', 120),
        ('Sweet Potato', 'Owerri', 'Imo', 'March', 'May', 'August', 'October', 'March', 'November', 120),
        ('Sweet Potato', 'Benin City', 'Edo', 'March', 'May', 'August', 'October', 'March', 'November', 120),
        
        # COCOYAM
        ('Cocoyam', 'Lagos', 'Lagos', 'March', 'May', 'September', 'November', 'March', 'November', 240),
        ('Cocoyam', 'Enugu', 'Enugu', 'March', 'May', 'September', 'November', 'March', 'November', 240),
        ('Cocoyam', 'Owerri', 'Imo', 'March', 'May', 'September', 'November', 'March', 'November', 240),
        ('Cocoyam', 'Benin City', 'Edo', 'March', 'May', 'September', 'November', 'March', 'November', 240),
        ('Cocoyam', 'Calabar', 'Cross River', 'March', 'May', 'September', 'November', 'March', 'December', 240),
        ('Cocoyam', 'Port Harcourt', 'Rivers', 'March', 'May', 'September', 'November', 'March', 'December', 240),
        
        # PLANTAIN
        ('Plantain', 'Lagos', 'Lagos', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Plantain', 'Enugu', 'Enugu', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Plantain', 'Owerri', 'Imo', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Plantain', 'Benin City', 'Edo', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Plantain', 'Calabar', 'Cross River', 'January', 'March', 'September', 'November', 'March', 'December', 365),
        ('Plantain', 'Port Harcourt', 'Rivers', 'January', 'March', 'September', 'November', 'March', 'December', 365),
        ('Plantain', 'Ibadan', 'Oyo', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        
        # COTTON
        ('Cotton', 'Kano', 'Kano', 'June', 'July', 'October', 'December', 'June', 'September', 150),
        ('Cotton', 'Kaduna', 'Kaduna', 'June', 'July', 'October', 'December', 'May', 'October', 150),
        ('Cotton', 'Katsina', 'Katsina', 'June', 'July', 'October', 'December', 'June', 'September', 150),
        ('Cotton', 'Sokoto', 'Sokoto', 'June', 'July', 'October', 'December', 'July', 'September', 150),
        ('Cotton', 'Zaria', 'Kaduna', 'June', 'July', 'October', 'December', 'May', 'October', 150),
        ('Cotton', 'Bauchi', 'Bauchi', 'June', 'July', 'October', 'December', 'July', 'September', 150),
        
        # SESAME
        ('Sesame', 'Kano', 'Kano', 'June', 'July', 'September', 'November', 'June', 'September', 90),
        ('Sesame', 'Kaduna', 'Kaduna', 'June', 'July', 'September', 'November', 'May', 'October', 90),
        ('Sesame', 'Sokoto', 'Sokoto', 'June', 'July', 'September', 'November', 'July', 'September', 90),
        ('Sesame', 'Katsina', 'Katsina', 'June', 'July', 'September', 'November', 'June', 'September', 90),
        ('Sesame', 'Bauchi', 'Bauchi', 'June', 'July', 'September', 'November', 'July', 'September', 90),
        ('Sesame', 'Gombe', 'Gombe', 'June', 'July', 'September', 'November', 'July', 'September', 90),
        
        # MELON (EGUSI)
        ('Melon', 'Lagos', 'Lagos', 'March', 'May', 'July', 'September', 'March', 'November', 90),
        ('Melon', 'Ibadan', 'Oyo', 'March', 'May', 'July', 'September', 'March', 'November', 90),
        ('Melon', 'Ilorin', 'Kwara', 'April', 'May', 'August', 'October', 'April', 'October', 90),
        ('Melon', 'Abuja', 'FCT', 'April', 'May', 'August', 'October', 'April', 'October', 90),
        ('Melon', 'Kaduna', 'Kaduna', 'May', 'June', 'August', 'October', 'May', 'October', 90),
        
        # OKRA
        ('Okra', 'Lagos', 'Lagos', 'March', 'May', 'June', 'September', 'March', 'November', 75),
        ('Okra', 'Ibadan', 'Oyo', 'March', 'May', 'June', 'September', 'March', 'November', 75),
        ('Okra', 'Enugu', 'Enugu', 'March', 'May', 'June', 'September', 'March', 'November', 75),
        ('Okra', 'Benin City', 'Edo', 'March', 'May', 'June', 'September', 'March', 'November', 75),
        ('Okra', 'Port Harcourt', 'Rivers', 'March', 'May', 'June', 'September', 'March', 'December', 75),
        ('Okra', 'Abuja', 'FCT', 'April', 'May', 'July', 'September', 'April', 'October', 75),
        ('Okra', 'Kano', 'Kano', 'May', 'June', 'July', 'September', 'June', 'September', 75),
        
        # CABBAGE
        ('Cabbage', 'Jos', 'Plateau', 'September', 'November', 'January', 'March', 'April', 'October', 90),
        ('Cabbage', 'Kano', 'Kano', 'October', 'December', 'February', 'April', 'June', 'September', 90),
        ('Cabbage', 'Kaduna', 'Kaduna', 'September', 'November', 'January', 'March', 'May', 'October', 90),
        ('Cabbage', 'Abuja', 'FCT', 'September', 'November', 'January', 'March', 'April', 'October', 90),
        ('Cabbage', 'Ibadan', 'Oyo', 'October', 'December', 'February', 'April', 'March', 'November', 90),
        
        # ONION
        ('Onion', 'Kano', 'Kano', 'October', 'December', 'February', 'April', 'June', 'September', 120),
        ('Onion', 'Sokoto', 'Sokoto', 'October', 'December', 'February', 'April', 'July', 'September', 120),
        ('Onion', 'Kaduna', 'Kaduna', 'October', 'December', 'February', 'April', 'May', 'October', 120),
        ('Onion', 'Katsina', 'Katsina', 'October', 'December', 'February', 'April', 'June', 'September', 120),
        ('Onion', 'Jos', 'Plateau', 'September', 'November', 'January', 'March', 'April', 'October', 120),
        
        # IRISH POTATO
        ('Irish Potato', 'Jos', 'Plateau', 'March', 'May', 'August', 'October', 'April', 'October', 120),
        ('Irish Potato', 'Jos', 'Plateau', 'August', 'October', 'December', 'February', 'April', 'October', 120),
        ('Irish Potato', 'Kaduna', 'Kaduna', 'March', 'May', 'August', 'October', 'May', 'October', 120),
        ('Irish Potato', 'Abuja', 'FCT', 'March', 'May', 'August', 'October', 'April', 'October', 120),
        
        # WHEAT
        ('Wheat', 'Kano', 'Kano', 'November', 'December', 'March', 'April', 'June', 'September', 110),
        ('Wheat', 'Kaduna', 'Kaduna', 'November', 'December', 'March', 'April', 'May', 'October', 110),
        ('Wheat', 'Jos', 'Plateau', 'November', 'December', 'March', 'April', 'April', 'October', 110),
        ('Wheat', 'Sokoto', 'Sokoto', 'November', 'December', 'March', 'April', 'July', 'September', 110),
        
        # CASHEW
        ('Cashew', 'Enugu', 'Enugu', 'February', 'April', 'October', 'December', 'March', 'November', 270),
        ('Cashew', 'Owerri', 'Imo', 'February', 'April', 'October', 'December', 'March', 'November', 270),
        ('Cashew', 'Benin City', 'Edo', 'February', 'April', 'October', 'December', 'March', 'November', 270),
        ('Cashew', 'Ibadan', 'Oyo', 'February', 'April', 'October', 'December', 'March', 'November', 270),
        ('Cashew', 'Akure', 'Ondo', 'February', 'April', 'October', 'December', 'March', 'November', 270),
        
        # OIL PALM
        ('Oil Palm', 'Lagos', 'Lagos', 'January', 'March', 'October', 'December', 'March', 'November', 365),
        ('Oil Palm', 'Port Harcourt', 'Rivers', 'January', 'March', 'October', 'December', 'March', 'December', 365),
        ('Oil Palm', 'Benin City', 'Edo', 'January', 'March', 'October', 'December', 'March', 'November', 365),
        ('Oil Palm', 'Enugu', 'Enugu', 'January', 'March', 'October', 'December', 'March', 'November', 365),
        ('Oil Palm', 'Calabar', 'Cross River', 'January', 'March', 'October', 'December', 'March', 'December', 365),
        
        # COCOA
        ('Cocoa', 'Ibadan', 'Oyo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Cocoa', 'Akure', 'Ondo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Cocoa', 'Benin City', 'Edo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Cocoa', 'Calabar', 'Cross River', 'March', 'May', 'September', 'November', 'March', 'December', 365),
        ('Cocoa', 'Owerri', 'Imo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        
        # COFFEE
        ('Coffee', 'Jos', 'Plateau', 'March', 'May', 'September', 'November', 'April', 'October', 365),
        ('Coffee', 'Ibadan', 'Oyo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Coffee', 'Akure', 'Ondo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        
        # BANANA
        ('Banana', 'Lagos', 'Lagos', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Banana', 'Port Harcourt', 'Rivers', 'January', 'March', 'September', 'November', 'March', 'December', 365),
        ('Banana', 'Enugu', 'Enugu', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Banana', 'Benin City', 'Edo', 'January', 'March', 'September', 'November', 'March', 'November', 365),
        ('Banana', 'Calabar', 'Cross River', 'January', 'March', 'September', 'November', 'March', 'December', 365),
        
        # CARROT
        ('Carrot', 'Jos', 'Plateau', 'September', 'November', 'January', 'March', 'April', 'October', 100),
        ('Carrot', 'Kano', 'Kano', 'October', 'December', 'February', 'April', 'June', 'September', 100),
        ('Carrot', 'Kaduna', 'Kaduna', 'September', 'November', 'January', 'March', 'May', 'October', 100),
        ('Carrot', 'Abuja', 'FCT', 'September', 'November', 'January', 'March', 'April', 'October', 100),
        
        # CUCUMBER
        ('Cucumber', 'Lagos', 'Lagos', 'March', 'May', 'June', 'August', 'March', 'November', 60),
        ('Cucumber', 'Ibadan', 'Oyo', 'March', 'May', 'June', 'August', 'March', 'November', 60),
        ('Cucumber', 'Enugu', 'Enugu', 'March', 'May', 'June', 'August', 'March', 'November', 60),
        ('Cucumber', 'Abuja', 'FCT', 'April', 'May', 'July', 'September', 'April', 'October', 60),
        ('Cucumber', 'Kano', 'Kano', 'May', 'June', 'July', 'September', 'June', 'September', 60),
        
        # LETTUCE
        ('Lettuce', 'Jos', 'Plateau', 'September', 'November', 'December', 'February', 'April', 'October', 60),
        ('Lettuce', 'Kano', 'Kano', 'October', 'December', 'January', 'March', 'June', 'September', 60),
        ('Lettuce', 'Abuja', 'FCT', 'September', 'November', 'December', 'February', 'April', 'October', 60),
        ('Lettuce', 'Ibadan', 'Oyo', 'October', 'December', 'January', 'March', 'March', 'November', 60),
        
        # WATERMELON
        ('Watermelon', 'Lagos', 'Lagos', 'March', 'May', 'June', 'August', 'March', 'November', 80),
        ('Watermelon', 'Ibadan', 'Oyo', 'March', 'May', 'June', 'August', 'March', 'November', 80),
        ('Watermelon', 'Kano', 'Kano', 'May', 'June', 'August', 'October', 'June', 'September', 80),
        ('Watermelon', 'Kaduna', 'Kaduna', 'May', 'June', 'August', 'October', 'May', 'October', 80),
        ('Watermelon', 'Abuja', 'FCT', 'April', 'May', 'July', 'September', 'April', 'October', 80),
        
        # PINEAPPLE
        ('Pineapple', 'Lagos', 'Lagos', 'March', 'May', 'September', 'November', 'March', 'November', 480),
        ('Pineapple', 'Benin City', 'Edo', 'March', 'May', 'September', 'November', 'March', 'November', 480),
        ('Pineapple', 'Ibadan', 'Oyo', 'March', 'May', 'September', 'November', 'March', 'November', 480),
        ('Pineapple', 'Enugu', 'Enugu', 'March', 'May', 'September', 'November', 'March', 'November', 480),
        ('Pineapple', 'Port Harcourt', 'Rivers', 'March', 'May', 'September', 'November', 'March', 'December', 480),
        
        # MANGO
        ('Mango', 'Kano', 'Kano', 'March', 'May', 'August', 'October', 'June', 'September', 365),
        ('Mango', 'Kaduna', 'Kaduna', 'March', 'May', 'August', 'October', 'May', 'October', 365),
        ('Mango', 'Abuja', 'FCT', 'March', 'May', 'August', 'October', 'April', 'October', 365),
        ('Mango', 'Ibadan', 'Oyo', 'March', 'May', 'August', 'October', 'March', 'November', 365),
        ('Mango', 'Jos', 'Plateau', 'March', 'May', 'August', 'October', 'April', 'October', 365),
        
        # ORANGE
        ('Orange', 'Ibadan', 'Oyo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Orange', 'Akure', 'Ondo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Orange', 'Benin City', 'Edo', 'March', 'May', 'September', 'November', 'March', 'November', 365),
        ('Orange', 'Jos', 'Plateau', 'March', 'May', 'September', 'November', 'April', 'October', 365),
        ('Orange', 'Kaduna', 'Kaduna', 'March', 'May', 'September', 'November', 'May', 'October', 365)
    ]
    

    for s in default_seasons:
        conn.execute('''INSERT OR IGNORE INTO seasonal_calendar 
            (crop_type, region, state, planting_start, planting_end, harvest_start, harvest_end, rainy_season_start, rainy_season_end, growing_days) 
            VALUES (?,?,?,?,?,?,?,?,?,?)''', s)

    # ============================================
    # ✅ 9. CROP NUTRIENT REQUIREMENTS BY REGION
    # ============================================
    conn.execute('''CREATE TABLE IF NOT EXISTS crop_nutrient_requirements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop_type TEXT NOT NULL,
        region TEXT NOT NULL,
        state TEXT,
        n_requirement REAL,
        p_requirement REAL,
        k_requirement REAL,
        optimal_ph_min REAL,
        optimal_ph_max REAL,
        growth_stage TEXT,
        planting_season TEXT,
        UNIQUE(crop_type, region, growth_stage)
    )''')

    default_crop_requirements = [
    # ==========================================
    # MAIZE Requirements (All Regions)
    # ==========================================
    ('Maize', 'Lagos', 'Lagos', 120, 60, 80, 5.5, 7.0, 'vegetative', 'March-April'),
    ('Maize', 'Lagos', 'Lagos', 150, 80, 100, 5.5, 7.0, 'flowering', 'March-April'),
    ('Maize', 'Lagos', 'Lagos', 100, 70, 90, 5.5, 7.0, 'fruiting', 'March-April'),
    
    ('Maize', 'Abuja', 'FCT', 110, 55, 75, 5.8, 7.2, 'vegetative', 'April-May'),
    ('Maize', 'Abuja', 'FCT', 140, 75, 95, 5.8, 7.2, 'flowering', 'April-May'),
    ('Maize', 'Abuja', 'FCT', 95, 65, 85, 5.8, 7.2, 'fruiting', 'April-May'),
    
    ('Maize', 'Kano', 'Kano', 100, 50, 70, 6.0, 7.5, 'vegetative', 'May-June'),
    ('Maize', 'Kano', 'Kano', 130, 70, 90, 6.0, 7.5, 'flowering', 'May-June'),
    ('Maize', 'Kano', 'Kano', 90, 60, 80, 6.0, 7.5, 'fruiting', 'May-June'),
    
    ('Maize', 'Ibadan', 'Oyo', 115, 58, 78, 5.9, 7.1, 'vegetative', 'March-April'),
    ('Maize', 'Ibadan', 'Oyo', 145, 78, 98, 5.9, 7.1, 'flowering', 'March-April'),
    ('Maize', 'Ibadan', 'Oyo', 98, 68, 88, 5.9, 7.1, 'fruiting', 'March-April'),
    
    ('Maize', 'Port Harcourt', 'Rivers', 118, 60, 82, 5.7, 6.9, 'vegetative', 'March-April'),
    ('Maize', 'Port Harcourt', 'Rivers', 148, 80, 102, 5.7, 6.9, 'flowering', 'March-April'),
    ('Maize', 'Port Harcourt', 'Rivers', 100, 70, 92, 5.7, 6.9, 'fruiting', 'March-April'),
    
    ('Maize', 'Enugu', 'Enugu', 112, 56, 76, 5.8, 7.0, 'vegetative', 'March-April'),
    ('Maize', 'Enugu', 'Enugu', 142, 76, 96, 5.8, 7.0, 'flowering', 'March-April'),
    ('Maize', 'Enugu', 'Enugu', 96, 66, 86, 5.8, 7.0, 'fruiting', 'March-April'),
    
    ('Maize', 'Kaduna', 'Kaduna', 105, 52, 72, 6.0, 7.3, 'vegetative', 'May-June'),
    ('Maize', 'Kaduna', 'Kaduna', 135, 72, 92, 6.0, 7.3, 'flowering', 'May-June'),
    ('Maize', 'Kaduna', 'Kaduna', 92, 62, 82, 6.0, 7.3, 'fruiting', 'May-June'),
    
    ('Maize', 'Benin City', 'Edo', 114, 58, 80, 5.8, 7.0, 'vegetative', 'March-April'),
    ('Maize', 'Benin City', 'Edo', 144, 78, 100, 5.8, 7.0, 'flowering', 'March-April'),
    ('Maize', 'Benin City', 'Edo', 98, 68, 90, 5.8, 7.0, 'fruiting', 'March-April'),
    
    ('Maize', 'Maiduguri', 'Borno', 98, 48, 68, 6.2, 7.5, 'vegetative', 'June-July'),
    ('Maize', 'Maiduguri', 'Borno', 128, 68, 88, 6.2, 7.5, 'flowering', 'June-July'),
    ('Maize', 'Maiduguri', 'Borno', 88, 58, 78, 6.2, 7.5, 'fruiting', 'June-July'),
    
    # ==========================================
    # RICE Requirements (All Regions)
    # ==========================================
    ('Rice', 'Lagos', 'Lagos', 100, 50, 70, 5.5, 6.5, 'vegetative', 'April-May'),
    ('Rice', 'Lagos', 'Lagos', 130, 65, 85, 5.5, 6.5, 'flowering', 'April-May'),
    ('Rice', 'Lagos', 'Lagos', 90, 60, 80, 5.5, 6.5, 'fruiting', 'April-May'),
    
    ('Rice', 'Abuja', 'FCT', 95, 48, 68, 5.7, 6.7, 'vegetative', 'May-June'),
    ('Rice', 'Abuja', 'FCT', 125, 63, 83, 5.7, 6.7, 'flowering', 'May-June'),
    ('Rice', 'Abuja', 'FCT', 88, 58, 78, 5.7, 6.7, 'fruiting', 'May-June'),
    
    ('Rice', 'Kano', 'Kano', 90, 45, 65, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Rice', 'Kano', 'Kano', 120, 60, 80, 6.0, 7.0, 'flowering', 'June-July'),
    ('Rice', 'Kano', 'Kano', 85, 55, 75, 6.0, 7.0, 'fruiting', 'June-July'),
    
    ('Rice', 'Ibadan', 'Oyo', 98, 49, 69, 5.6, 6.6, 'vegetative', 'April-May'),
    ('Rice', 'Ibadan', 'Oyo', 128, 64, 84, 5.6, 6.6, 'flowering', 'April-May'),
    ('Rice', 'Ibadan', 'Oyo', 89, 59, 79, 5.6, 6.6, 'fruiting', 'April-May'),
    
    ('Rice', 'Port Harcourt', 'Rivers', 102, 51, 71, 5.5, 6.5, 'vegetative', 'April-May'),
    ('Rice', 'Port Harcourt', 'Rivers', 132, 66, 86, 5.5, 6.5, 'flowering', 'April-May'),
    ('Rice', 'Port Harcourt', 'Rivers', 91, 61, 81, 5.5, 6.5, 'fruiting', 'April-May'),
    
    ('Rice', 'Enugu', 'Enugu', 99, 50, 70, 5.6, 6.6, 'vegetative', 'April-May'),
    ('Rice', 'Enugu', 'Enugu', 129, 65, 85, 5.6, 6.6, 'flowering', 'April-May'),
    ('Rice', 'Enugu', 'Enugu', 90, 60, 80, 5.6, 6.6, 'fruiting', 'April-May'),
    
    ('Rice', 'Kaduna', 'Kaduna', 93, 47, 67, 5.9, 6.9, 'vegetative', 'May-June'),
    ('Rice', 'Kaduna', 'Kaduna', 123, 62, 82, 5.9, 6.9, 'flowering', 'May-June'),
    ('Rice', 'Kaduna', 'Kaduna', 87, 57, 77, 5.9, 6.9, 'fruiting', 'May-June'),
    
    ('Rice', 'Benin City', 'Edo', 100, 50, 70, 5.5, 6.5, 'vegetative', 'April-May'),
    ('Rice', 'Benin City', 'Edo', 130, 65, 85, 5.5, 6.5, 'flowering', 'April-May'),
    ('Rice', 'Benin City', 'Edo', 90, 60, 80, 5.5, 6.5, 'fruiting', 'April-May'),
    
    # ==========================================
    # TOMATO Requirements (All Regions)
    # ==========================================
    ('Tomato', 'Lagos', 'Lagos', 150, 80, 120, 6.0, 6.8, 'vegetative', 'October-November'),
    ('Tomato', 'Lagos', 'Lagos', 180, 100, 150, 6.0, 6.8, 'flowering', 'October-November'),
    ('Tomato', 'Lagos', 'Lagos', 140, 90, 130, 6.0, 6.8, 'fruiting', 'October-November'),
    
    ('Tomato', 'Kano', 'Kano', 140, 75, 115, 6.2, 7.0, 'vegetative', 'September-October'),
    ('Tomato', 'Kano', 'Kano', 170, 95, 145, 6.2, 7.0, 'flowering', 'September-October'),
    ('Tomato', 'Kano', 'Kano', 135, 85, 125, 6.2, 7.0, 'fruiting', 'September-October'),
    
    ('Tomato', 'Abuja', 'FCT', 145, 78, 118, 6.1, 6.9, 'vegetative', 'October-November'),
    ('Tomato', 'Abuja', 'FCT', 175, 98, 148, 6.1, 6.9, 'flowering', 'October-November'),
    ('Tomato', 'Abuja', 'FCT', 138, 88, 128, 6.1, 6.9, 'fruiting', 'October-November'),
    
    ('Tomato', 'Ibadan', 'Oyo', 148, 79, 119, 6.0, 6.8, 'vegetative', 'October-November'),
    ('Tomato', 'Ibadan', 'Oyo', 178, 99, 149, 6.0, 6.8, 'flowering', 'October-November'),
    ('Tomato', 'Ibadan', 'Oyo', 139, 89, 129, 6.0, 6.8, 'fruiting', 'October-November'),
    
    ('Tomato', 'Jos', 'Plateau', 142, 76, 116, 6.2, 7.0, 'vegetative', 'September-October'),
    ('Tomato', 'Jos', 'Plateau', 172, 96, 146, 6.2, 7.0, 'flowering', 'September-October'),
    ('Tomato', 'Jos', 'Plateau', 136, 86, 126, 6.2, 7.0, 'fruiting', 'September-October'),
    
    # ==========================================
    # CASSAVA Requirements (All Regions)
    # ==========================================
    ('Cassava', 'Lagos', 'Lagos', 80, 40, 100, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Cassava', 'Lagos', 'Lagos', 60, 30, 80, 5.5, 6.5, 'maturity', 'January-March'),
    
    ('Cassava', 'Abuja', 'FCT', 75, 38, 95, 5.7, 6.7, 'vegetative', 'February-April'),
    ('Cassava', 'Abuja', 'FCT', 58, 28, 78, 5.7, 6.7, 'maturity', 'February-April'),
    
    ('Cassava', 'Kano', 'Kano', 70, 35, 90, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Cassava', 'Kano', 'Kano', 55, 25, 75, 6.0, 7.0, 'maturity', 'March-May'),
    
    ('Cassava', 'Ibadan', 'Oyo', 78, 39, 98, 5.6, 6.6, 'vegetative', 'January-March'),
    ('Cassava', 'Ibadan', 'Oyo', 59, 29, 79, 5.6, 6.6, 'maturity', 'January-March'),
    
    ('Cassava', 'Port Harcourt', 'Rivers', 82, 41, 102, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Cassava', 'Port Harcourt', 'Rivers', 61, 31, 81, 5.5, 6.5, 'maturity', 'January-March'),
    
    ('Cassava', 'Enugu', 'Enugu', 79, 40, 99, 5.6, 6.6, 'vegetative', 'February-April'),
    ('Cassava', 'Enugu', 'Enugu', 60, 30, 80, 5.6, 6.6, 'maturity', 'February-April'),
    
    ('Cassava', 'Benin City', 'Edo', 80, 40, 100, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Cassava', 'Benin City', 'Edo', 60, 30, 80, 5.5, 6.5, 'maturity', 'January-March'),
    
    # ==========================================
    # PEPPER Requirements (All Regions)
    # ==========================================
    ('Pepper', 'Lagos', 'Lagos', 110, 60, 90, 6.0, 6.8, 'vegetative', 'September-October'),
    ('Pepper', 'Lagos', 'Lagos', 140, 80, 110, 6.0, 6.8, 'flowering', 'September-October'),
    ('Pepper', 'Lagos', 'Lagos', 100, 70, 100, 6.0, 6.8, 'fruiting', 'September-October'),
    
    ('Pepper', 'Kano', 'Kano', 105, 55, 85, 6.2, 7.0, 'vegetative', 'August-September'),
    ('Pepper', 'Kano', 'Kano', 135, 75, 105, 6.2, 7.0, 'flowering', 'August-September'),
    ('Pepper', 'Kano', 'Kano', 95, 65, 95, 6.2, 7.0, 'fruiting', 'August-September'),
    
    ('Pepper', 'Abuja', 'FCT', 108, 58, 88, 6.1, 6.9, 'vegetative', 'September-October'),
    ('Pepper', 'Abuja', 'FCT', 138, 78, 108, 6.1, 6.9, 'flowering', 'September-October'),
    ('Pepper', 'Abuja', 'FCT', 98, 68, 98, 6.1, 6.9, 'fruiting', 'September-October'),
    
    ('Pepper', 'Ibadan', 'Oyo', 109, 59, 89, 6.0, 6.8, 'vegetative', 'September-October'),
    ('Pepper', 'Ibadan', 'Oyo', 139, 79, 109, 6.0, 6.8, 'flowering', 'September-October'),
    ('Pepper', 'Ibadan', 'Oyo', 99, 69, 99, 6.0, 6.8, 'fruiting', 'September-October'),
    
    # ==========================================
    # YAM Requirements (All Regions)
    # ==========================================
    ('Yam', 'Lagos', 'Lagos', 90, 45, 110, 5.5, 6.5, 'vegetative', 'February-April'),
    ('Yam', 'Lagos', 'Lagos', 110, 55, 130, 5.5, 6.5, 'maturity', 'February-April'),
    
    ('Yam', 'Abuja', 'FCT', 85, 40, 105, 5.8, 6.8, 'vegetative', 'February-April'),
    ('Yam', 'Abuja', 'FCT', 105, 50, 125, 5.8, 6.8, 'maturity', 'February-April'),
    
    ('Yam', 'Kano', 'Kano', 80, 35, 100, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Yam', 'Kano', 'Kano', 100, 45, 120, 6.0, 7.0, 'maturity', 'March-May'),
    
    ('Yam', 'Ibadan', 'Oyo', 88, 43, 108, 5.7, 6.7, 'vegetative', 'February-April'),
    ('Yam', 'Ibadan', 'Oyo', 108, 53, 128, 5.7, 6.7, 'maturity', 'February-April'),
    
    ('Yam', 'Enugu', 'Enugu', 87, 42, 107, 5.7, 6.7, 'vegetative', 'February-April'),
    ('Yam', 'Enugu', 'Enugu', 107, 52, 127, 5.7, 6.7, 'maturity', 'February-April'),
    
    ('Yam', 'Benin City', 'Edo', 89, 44, 109, 5.6, 6.6, 'vegetative', 'February-April'),
    ('Yam', 'Benin City', 'Edo', 109, 54, 129, 5.6, 6.6, 'maturity', 'February-April'),
    
    # ==========================================
    # SORGHUM Requirements (Northern Regions)
    # ==========================================
    ('Sorghum', 'Kano', 'Kano', 80, 40, 60, 6.0, 7.5, 'vegetative', 'May-June'),
    ('Sorghum', 'Kano', 'Kano', 100, 50, 75, 6.0, 7.5, 'flowering', 'May-June'),
    ('Sorghum', 'Kano', 'Kano', 70, 35, 55, 6.0, 7.5, 'grain filling', 'May-June'),
    
    ('Sorghum', 'Kaduna', 'Kaduna', 75, 35, 55, 6.0, 7.0, 'vegetative', 'May-June'),
    ('Sorghum', 'Kaduna', 'Kaduna', 95, 45, 70, 6.0, 7.0, 'flowering', 'May-June'),
    ('Sorghum', 'Kaduna', 'Kaduna', 65, 30, 50, 6.0, 7.0, 'grain filling', 'May-June'),
    
    ('Sorghum', 'Abuja', 'FCT', 78, 38, 58, 5.9, 7.2, 'vegetative', 'May-June'),
    ('Sorghum', 'Abuja', 'FCT', 98, 48, 73, 5.9, 7.2, 'flowering', 'May-June'),
    ('Sorghum', 'Abuja', 'FCT', 68, 33, 53, 5.9, 7.2, 'grain filling', 'May-June'),
    
    # ==========================================
    # MILLET Requirements (Northern Regions)
    # ==========================================
    ('Millet', 'Kano', 'Kano', 60, 30, 50, 6.0, 7.5, 'vegetative', 'June-July'),
    ('Millet', 'Kano', 'Kano', 75, 38, 62, 6.0, 7.5, 'flowering', 'June-July'),
    ('Millet', 'Kano', 'Kano', 50, 25, 42, 6.0, 7.5, 'grain filling', 'June-July'),
    
    ('Millet', 'Kaduna', 'Kaduna', 58, 28, 48, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Millet', 'Kaduna', 'Kaduna', 73, 36, 60, 6.0, 7.0, 'flowering', 'June-July'),
    ('Millet', 'Kaduna', 'Kaduna', 48, 23, 40, 6.0, 7.0, 'grain filling', 'June-July'),
    
    ('Millet', 'Maiduguri', 'Borno', 55, 26, 46, 6.2, 7.5, 'vegetative', 'June-July'),
    ('Millet', 'Maiduguri', 'Borno', 70, 34, 58, 6.2, 7.5, 'flowering', 'June-July'),
    ('Millet', 'Maiduguri', 'Borno', 46, 21, 38, 6.2, 7.5, 'grain filling', 'June-July'),
    
    # ==========================================
    # GROUNDNUT Requirements (Northern Regions)
    # ==========================================
    ('Groundnut', 'Kano', 'Kano', 50, 60, 90, 5.8, 7.0, 'vegetative', 'June-July'),
    ('Groundnut', 'Kano', 'Kano', 65, 75, 110, 5.8, 7.0, 'flowering', 'June-July'),
    ('Groundnut', 'Kano', 'Kano', 45, 55, 85, 5.8, 7.0, 'pod filling', 'June-July'),
    
    ('Groundnut', 'Kaduna', 'Kaduna', 48, 58, 88, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Groundnut', 'Kaduna', 'Kaduna', 62, 72, 105, 6.0, 7.0, 'flowering', 'June-July'),
    ('Groundnut', 'Kaduna', 'Kaduna', 42, 52, 82, 6.0, 7.0, 'pod filling', 'June-July'),
    
    ('Groundnut', 'Abuja', 'FCT', 49, 59, 89, 5.9, 7.0, 'vegetative', 'June-July'),
    ('Groundnut', 'Abuja', 'FCT', 63, 73, 107, 5.9, 7.0, 'flowering', 'June-July'),
    ('Groundnut', 'Abuja', 'FCT', 43, 53, 83, 5.9, 7.0, 'pod filling', 'June-July'),
    
    # ==========================================
    # COWPEA Requirements (All Regions)
    # ==========================================
    ('Cowpea', 'Kano', 'Kano', 40, 50, 70, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Cowpea', 'Kano', 'Kano', 55, 65, 85, 6.0, 7.0, 'flowering', 'June-July'),
    ('Cowpea', 'Kano', 'Kano', 35, 45, 65, 6.0, 7.0, 'pod filling', 'June-July'),
    
    ('Cowpea', 'Kaduna', 'Kaduna', 38, 48, 68, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Cowpea', 'Kaduna', 'Kaduna', 52, 62, 82, 6.0, 7.0, 'flowering', 'June-July'),
    ('Cowpea', 'Kaduna', 'Kaduna', 32, 42, 62, 6.0, 7.0, 'pod filling', 'June-July'),
    
    ('Cowpea', 'Abuja', 'FCT', 39, 49, 69, 5.9, 7.0, 'vegetative', 'June-July'),
    ('Cowpea', 'Abuja', 'FCT', 53, 63, 83, 5.9, 7.0, 'flowering', 'June-July'),
    ('Cowpea', 'Abuja', 'FCT', 33, 43, 63, 5.9, 7.0, 'pod filling', 'June-July'),
    
    ('Cowpea', 'Ibadan', 'Oyo', 41, 51, 71, 5.8, 6.8, 'vegetative', 'April-May'),
    ('Cowpea', 'Ibadan', 'Oyo', 56, 66, 86, 5.8, 6.8, 'flowering', 'April-May'),
    ('Cowpea', 'Ibadan', 'Oyo', 36, 46, 66, 5.8, 6.8, 'pod filling', 'April-May'),
    
    # ==========================================
    # SOYBEAN Requirements (All Regions)
    # ==========================================
    ('Soybean', 'Abuja', 'FCT', 60, 70, 80, 6.0, 7.0, 'vegetative', 'May-June'),
    ('Soybean', 'Abuja', 'FCT', 75, 85, 95, 6.0, 7.0, 'flowering', 'May-June'),
    ('Soybean', 'Abuja', 'FCT', 55, 65, 75, 6.0, 7.0, 'pod filling', 'May-June'),
    
    ('Soybean', 'Kaduna', 'Kaduna', 58, 68, 78, 6.0, 7.0, 'vegetative', 'May-June'),
    ('Soybean', 'Kaduna', 'Kaduna', 72, 82, 92, 6.0, 7.0, 'flowering', 'May-June'),
    ('Soybean', 'Kaduna', 'Kaduna', 52, 62, 72, 6.0, 7.0, 'pod filling', 'May-June'),
    
    ('Soybean', 'Ibadan', 'Oyo', 61, 71, 81, 5.8, 6.8, 'vegetative', 'April-May'),
    ('Soybean', 'Ibadan', 'Oyo', 76, 86, 96, 5.8, 6.8, 'flowering', 'April-May'),
    ('Soybean', 'Ibadan', 'Oyo', 56, 66, 76, 5.8, 6.8, 'pod filling', 'April-May'),
    
    # ==========================================
    # SWEET POTATO Requirements (All Regions)
    # ==========================================
    ('Sweet Potato', 'Lagos', 'Lagos', 70, 50, 100, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Sweet Potato', 'Lagos', 'Lagos', 85, 60, 115, 5.5, 6.5, 'tuber bulking', 'March-May'),
    ('Sweet Potato', 'Lagos', 'Lagos', 65, 45, 95, 5.5, 6.5, 'maturity', 'March-May'),
    
    ('Sweet Potato', 'Abuja', 'FCT', 68, 48, 98, 5.8, 6.8, 'vegetative', 'March-May'),
    ('Sweet Potato', 'Abuja', 'FCT', 82, 58, 112, 5.8, 6.8, 'tuber bulking', 'March-May'),
    ('Sweet Potato', 'Abuja', 'FCT', 62, 42, 92, 5.8, 6.8, 'maturity', 'March-May'),
    
    ('Sweet Potato', 'Ibadan', 'Oyo', 69, 49, 99, 5.7, 6.7, 'vegetative', 'March-May'),
    ('Sweet Potato', 'Ibadan', 'Oyo', 83, 59, 113, 5.7, 6.7, 'tuber bulking', 'March-May'),
    ('Sweet Potato', 'Ibadan', 'Oyo', 63, 43, 93, 5.7, 6.7, 'maturity', 'March-May'),
    
    # ==========================================
    # COCOYAM Requirements (Southern Regions)
    # ==========================================
    ('Cocoyam', 'Lagos', 'Lagos', 85, 45, 105, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Cocoyam', 'Lagos', 'Lagos', 100, 55, 120, 5.5, 6.5, 'corm development', 'March-May'),
    
    ('Cocoyam', 'Enugu', 'Enugu', 82, 42, 102, 5.8, 6.8, 'vegetative', 'March-May'),
    ('Cocoyam', 'Enugu', 'Enugu', 98, 52, 118, 5.8, 6.8, 'corm development', 'March-May'),
    
    ('Cocoyam', 'Port Harcourt', 'Rivers', 86, 46, 106, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Cocoyam', 'Port Harcourt', 'Rivers', 101, 56, 121, 5.5, 6.5, 'corm development', 'March-May'),
    
    ('Cocoyam', 'Benin City', 'Edo', 84, 44, 104, 5.6, 6.6, 'vegetative', 'March-May'),
    ('Cocoyam', 'Benin City', 'Edo', 99, 54, 119, 5.6, 6.6, 'corm development', 'March-May'),
    
    # ==========================================
    # PLANTAIN Requirements (Southern Regions)
    # ==========================================
    ('Plantain', 'Lagos', 'Lagos', 130, 70, 150, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Plantain', 'Lagos', 'Lagos', 150, 80, 170, 5.5, 6.5, 'flowering', 'January-March'),
    ('Plantain', 'Lagos', 'Lagos', 120, 65, 145, 5.5, 6.5, 'fruiting', 'January-March'),
    
    ('Plantain', 'Enugu', 'Enugu', 128, 68, 148, 5.8, 6.8, 'vegetative', 'January-March'),
    ('Plantain', 'Enugu', 'Enugu', 148, 78, 168, 5.8, 6.8, 'flowering', 'January-March'),
    ('Plantain', 'Enugu', 'Enugu', 118, 62, 142, 5.8, 6.8, 'fruiting', 'January-March'),
    
    ('Plantain', 'Port Harcourt', 'Rivers', 131, 71, 151, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Plantain', 'Port Harcourt', 'Rivers', 151, 81, 171, 5.5, 6.5, 'flowering', 'January-March'),
    ('Plantain', 'Port Harcourt', 'Rivers', 121, 66, 146, 5.5, 6.5, 'fruiting', 'January-March'),
    
    ('Plantain', 'Benin City', 'Edo', 129, 69, 149, 5.6, 6.6, 'vegetative', 'January-March'),
    ('Plantain', 'Benin City', 'Edo', 149, 79, 169, 5.6, 6.6, 'flowering', 'January-March'),
    ('Plantain', 'Benin City', 'Edo', 119, 64, 144, 5.6, 6.6, 'fruiting', 'January-March'),
    
    # ==========================================
    # COTTON Requirements (Northern Regions)
    # ==========================================
    ('Cotton', 'Kano', 'Kano', 100, 50, 90, 6.0, 7.5, 'vegetative', 'June-July'),
    ('Cotton', 'Kano', 'Kano', 120, 60, 110, 6.0, 7.5, 'flowering', 'June-July'),
    ('Cotton', 'Kano', 'Kano', 90, 45, 85, 6.0, 7.5, 'boll development', 'June-July'),
    
    ('Cotton', 'Kaduna', 'Kaduna', 98, 48, 88, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Cotton', 'Kaduna', 'Kaduna', 118, 58, 108, 6.0, 7.0, 'flowering', 'June-July'),
    ('Cotton', 'Kaduna', 'Kaduna', 88, 42, 82, 6.0, 7.0, 'boll development', 'June-July'),
    
    # ==========================================
    # SESAME Requirements (Northern Regions)
    # ==========================================
    ('Sesame', 'Kano', 'Kano', 55, 45, 65, 6.0, 7.5, 'vegetative', 'June-July'),
    ('Sesame', 'Kano', 'Kano', 70, 55, 80, 6.0, 7.5, 'flowering', 'June-July'),
    ('Sesame', 'Kano', 'Kano', 50, 40, 60, 6.0, 7.5, 'capsule filling', 'June-July'),
    
    ('Sesame', 'Kaduna', 'Kaduna', 52, 42, 62, 6.0, 7.0, 'vegetative', 'June-July'),
    ('Sesame', 'Kaduna', 'Kaduna', 68, 52, 78, 6.0, 7.0, 'flowering', 'June-July'),
    ('Sesame', 'Kaduna', 'Kaduna', 48, 38, 58, 6.0, 7.0, 'capsule filling', 'June-July'),
    
    # ==========================================
    # MELON (EGUSI) Requirements
    # ==========================================
    ('Melon', 'Lagos', 'Lagos', 75, 55, 85, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Melon', 'Lagos', 'Lagos', 90, 65, 100, 6.0, 7.0, 'flowering', 'March-May'),
    ('Melon', 'Lagos', 'Lagos', 70, 50, 80, 6.0, 7.0, 'fruit development', 'March-May'),
    
    ('Melon', 'Ibadan', 'Oyo', 72, 52, 82, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Melon', 'Ibadan', 'Oyo', 88, 62, 98, 6.0, 7.0, 'flowering', 'March-May'),
    ('Melon', 'Ibadan', 'Oyo', 68, 48, 78, 6.0, 7.0, 'fruit development', 'March-May'),
    
    ('Melon', 'Abuja', 'FCT', 73, 53, 83, 6.0, 7.0, 'vegetative', 'April-May'),
    ('Melon', 'Abuja', 'FCT', 89, 63, 99, 6.0, 7.0, 'flowering', 'April-May'),
    ('Melon', 'Abuja', 'FCT', 69, 49, 79, 6.0, 7.0, 'fruit development', 'April-May'),
    
    # ==========================================
    # OKRA Requirements (All Regions)
    # ==========================================
    ('Okra', 'Lagos', 'Lagos', 95, 55, 85, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Okra', 'Lagos', 'Lagos', 110, 65, 100, 6.0, 7.0, 'flowering', 'March-May'),
    ('Okra', 'Lagos', 'Lagos', 90, 50, 80, 6.0, 7.0, 'pod development', 'March-May'),
    
    ('Okra', 'Enugu', 'Enugu', 92, 52, 82, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Okra', 'Enugu', 'Enugu', 108, 62, 98, 6.0, 7.0, 'flowering', 'March-May'),
    ('Okra', 'Enugu', 'Enugu', 88, 48, 78, 6.0, 7.0, 'pod development', 'March-May'),
    
    ('Okra', 'Abuja', 'FCT', 93, 53, 83, 6.0, 7.0, 'vegetative', 'April-May'),
    ('Okra', 'Abuja', 'FCT', 109, 63, 99, 6.0, 7.0, 'flowering', 'April-May'),
    ('Okra', 'Abuja', 'FCT', 89, 49, 79, 6.0, 7.0, 'pod development', 'April-May'),
    
    # ==========================================
    # CABBAGE Requirements (Cooler Regions)
    # ==========================================
    ('Cabbage', 'Jos', 'Plateau', 140, 70, 110, 6.0, 7.0, 'vegetative', 'September-November'),
    ('Cabbage', 'Jos', 'Plateau', 160, 80, 130, 6.0, 7.0, 'head formation', 'September-November'),
    
    ('Cabbage', 'Kano', 'Kano', 138, 68, 108, 6.2, 7.2, 'vegetative', 'October-December'),
    ('Cabbage', 'Kano', 'Kano', 158, 78, 128, 6.2, 7.2, 'head formation', 'October-December'),
    
    ('Cabbage', 'Abuja', 'FCT', 139, 69, 109, 6.1, 7.1, 'vegetative', 'September-November'),
    ('Cabbage', 'Abuja', 'FCT', 159, 79, 129, 6.1, 7.1, 'head formation', 'September-November'),
    
    # ==========================================
    # ONION Requirements (Northern Regions)
    # ==========================================
    ('Onion', 'Kano', 'Kano', 110, 65, 95, 6.0, 7.5, 'vegetative', 'October-December'),
    ('Onion', 'Kano', 'Kano', 130, 75, 115, 6.0, 7.5, 'bulb formation', 'October-December'),
    ('Onion', 'Kano', 'Kano', 105, 60, 90, 6.0, 7.5, 'maturity', 'October-December'),
    
    ('Onion', 'Kaduna', 'Kaduna', 108, 62, 92, 6.0, 7.0, 'vegetative', 'October-December'),
    ('Onion', 'Kaduna', 'Kaduna', 128, 72, 112, 6.0, 7.0, 'bulb formation', 'October-December'),
    ('Onion', 'Kaduna', 'Kaduna', 102, 58, 88, 6.0, 7.0, 'maturity', 'October-December'),
    
    ('Onion', 'Jos', 'Plateau', 109, 63, 93, 6.0, 7.0, 'vegetative', 'September-November'),
    ('Onion', 'Jos', 'Plateau', 129, 73, 113, 6.0, 7.0, 'bulb formation', 'September-November'),
    ('Onion', 'Jos', 'Plateau', 103, 59, 89, 6.0, 7.0, 'maturity', 'September-November'),
    
    # ==========================================
    # IRISH POTATO Requirements (Cooler Regions)
    # ==========================================
    ('Irish Potato', 'Jos', 'Plateau', 120, 75, 130, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Irish Potato', 'Jos', 'Plateau', 140, 85, 150, 5.5, 6.5, 'tuber initiation', 'March-May'),
    ('Irish Potato', 'Jos', 'Plateau', 115, 70, 125, 5.5, 6.5, 'tuber bulking', 'March-May'),
    
    ('Irish Potato', 'Kaduna', 'Kaduna', 118, 72, 128, 5.8, 6.8, 'vegetative', 'March-May'),
    ('Irish Potato', 'Kaduna', 'Kaduna', 138, 82, 148, 5.8, 6.8, 'tuber initiation', 'March-May'),
    ('Irish Potato', 'Kaduna', 'Kaduna', 112, 68, 122, 5.8, 6.8, 'tuber bulking', 'March-May'),
    
    # ==========================================
    # WHEAT Requirements (Northern Regions)
    # ==========================================
    ('Wheat', 'Kano', 'Kano', 90, 50, 70, 6.0, 7.5, 'vegetative', 'November-December'),
    ('Wheat', 'Kano', 'Kano', 110, 60, 85, 6.0, 7.5, 'tillering', 'November-December'),
    ('Wheat', 'Kano', 'Kano', 85, 45, 65, 6.0, 7.5, 'grain filling', 'November-December'),
    
    ('Wheat', 'Kaduna', 'Kaduna', 88, 48, 68, 6.0, 7.0, 'vegetative', 'November-December'),
    ('Wheat', 'Kaduna', 'Kaduna', 108, 58, 82, 6.0, 7.0, 'tillering', 'November-December'),
    ('Wheat', 'Kaduna', 'Kaduna', 82, 42, 62, 6.0, 7.0, 'grain filling', 'November-December'),
    
    # ==========================================
    # CASHEW Requirements (Southern Regions)
    # ==========================================
    ('Cashew', 'Enugu', 'Enugu', 70, 40, 80, 5.5, 6.5, 'vegetative', 'February-April'),
    ('Cashew', 'Enugu', 'Enugu', 85, 50, 95, 5.5, 6.5, 'flowering', 'February-April'),
    ('Cashew', 'Enugu', 'Enugu', 65, 35, 75, 5.5, 6.5, 'nut development', 'February-April'),
    
    ('Cashew', 'Ibadan', 'Oyo', 68, 38, 78, 5.8, 6.8, 'vegetative', 'February-April'),
    ('Cashew', 'Ibadan', 'Oyo', 82, 48, 92, 5.8, 6.8, 'flowering', 'February-April'),
    ('Cashew', 'Ibadan', 'Oyo', 62, 32, 72, 5.8, 6.8, 'nut development', 'February-April'),
    
    # ==========================================
    # OIL PALM Requirements (Southern Regions)
    # ==========================================
    ('Oil Palm', 'Lagos', 'Lagos', 100, 60, 120, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Oil Palm', 'Lagos', 'Lagos', 120, 70, 140, 5.5, 6.5, 'flowering', 'January-March'),
    ('Oil Palm', 'Lagos', 'Lagos', 95, 55, 115, 5.5, 6.5, 'fruit development', 'January-March'),
    
    ('Oil Palm', 'Port Harcourt', 'Rivers', 98, 58, 118, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Oil Palm', 'Port Harcourt', 'Rivers', 118, 68, 138, 5.5, 6.5, 'flowering', 'January-March'),
    ('Oil Palm', 'Port Harcourt', 'Rivers', 92, 52, 112, 5.5, 6.5, 'fruit development', 'January-March'),
    
    ('Oil Palm', 'Benin City', 'Edo', 99, 59, 119, 5.6, 6.6, 'vegetative', 'January-March'),
    ('Oil Palm', 'Benin City', 'Edo', 119, 69, 139, 5.6, 6.6, 'flowering', 'January-March'),
    ('Oil Palm', 'Benin City', 'Edo', 93, 53, 113, 5.6, 6.6, 'fruit development', 'January-March'),
    
    # ==========================================
    # COCOA Requirements (Southern Regions)
    # ==========================================
    ('Cocoa', 'Ibadan', 'Oyo', 85, 55, 95, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Cocoa', 'Ibadan', 'Oyo', 100, 65, 110, 5.5, 6.5, 'flowering', 'March-May'),
    ('Cocoa', 'Ibadan', 'Oyo', 80, 50, 90, 5.5, 6.5, 'pod development', 'March-May'),
    
    ('Cocoa', 'Akure', 'Ondo', 82, 52, 92, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Cocoa', 'Akure', 'Ondo', 98, 62, 108, 5.5, 6.5, 'flowering', 'March-May'),
    ('Cocoa', 'Akure', 'Ondo', 78, 48, 88, 5.5, 6.5, 'pod development', 'March-May'),
    
    ('Cocoa', 'Benin City', 'Edo', 84, 54, 94, 5.6, 6.6, 'vegetative', 'March-May'),
    ('Cocoa', 'Benin City', 'Edo', 99, 64, 109, 5.6, 6.6, 'flowering', 'March-May'),
    ('Cocoa', 'Benin City', 'Edo', 79, 49, 89, 5.6, 6.6, 'pod development', 'March-May'),
    
    # ==========================================
    # COFFEE Requirements (Southern Highlands)
    # ==========================================
    ('Coffee', 'Jos', 'Plateau', 90, 50, 85, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Coffee', 'Jos', 'Plateau', 105, 60, 100, 5.5, 6.5, 'flowering', 'March-May'),
    ('Coffee', 'Jos', 'Plateau', 85, 45, 80, 5.5, 6.5, 'berry development', 'March-May'),
    
    ('Coffee', 'Ibadan', 'Oyo', 88, 48, 82, 5.8, 6.8, 'vegetative', 'March-May'),
    ('Coffee', 'Ibadan', 'Oyo', 102, 58, 98, 5.8, 6.8, 'flowering', 'March-May'),
    ('Coffee', 'Ibadan', 'Oyo', 82, 42, 78, 5.8, 6.8, 'berry development', 'March-May'),
    
    # ==========================================
    # BANANA Requirements (Southern Regions)
    # ==========================================
    ('Banana', 'Lagos', 'Lagos', 125, 65, 145, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Banana', 'Lagos', 'Lagos', 145, 75, 165, 5.5, 6.5, 'flowering', 'January-March'),
    ('Banana', 'Lagos', 'Lagos', 120, 60, 140, 5.5, 6.5, 'fruit development', 'January-March'),
    
    ('Banana', 'Port Harcourt', 'Rivers', 122, 62, 142, 5.5, 6.5, 'vegetative', 'January-March'),
    ('Banana', 'Port Harcourt', 'Rivers', 142, 72, 162, 5.5, 6.5, 'flowering', 'January-March'),
    ('Banana', 'Port Harcourt', 'Rivers', 118, 58, 138, 5.5, 6.5, 'fruit development', 'January-March'),
    
    ('Banana', 'Enugu', 'Enugu', 124, 64, 144, 5.8, 6.8, 'vegetative', 'January-March'),
    ('Banana', 'Enugu', 'Enugu', 144, 74, 164, 5.8, 6.8, 'flowering', 'January-March'),
    ('Banana', 'Enugu', 'Enugu', 119, 59, 139, 5.8, 6.8, 'fruit development', 'January-March'),
    
    # ==========================================
    # CARROT Requirements (Cooler Regions)
    # ==========================================
    ('Carrot', 'Jos', 'Plateau', 85, 60, 95, 6.0, 7.0, 'vegetative', 'September-November'),
    ('Carrot', 'Jos', 'Plateau', 100, 70, 110, 6.0, 7.0, 'root development', 'September-November'),
    
    ('Carrot', 'Kano', 'Kano', 82, 58, 92, 6.2, 7.2, 'vegetative', 'October-December'),
    ('Carrot', 'Kano', 'Kano', 98, 68, 108, 6.2, 7.2, 'root development', 'October-December'),
    
    ('Carrot', 'Abuja', 'FCT', 83, 59, 93, 6.1, 7.1, 'vegetative', 'September-November'),
    ('Carrot', 'Abuja', 'FCT', 99, 69, 109, 6.1, 7.1, 'root development', 'September-November'),
    
    # ==========================================
    # CUCUMBER Requirements (All Regions)
    # ==========================================
    ('Cucumber', 'Lagos', 'Lagos', 90, 55, 85, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Cucumber', 'Lagos', 'Lagos', 105, 65, 100, 6.0, 7.0, 'flowering', 'March-May'),
    ('Cucumber', 'Lagos', 'Lagos', 85, 50, 80, 6.0, 7.0, 'fruit development', 'March-May'),
    
    ('Cucumber', 'Kano', 'Kano', 88, 52, 82, 6.2, 7.2, 'vegetative', 'May-June'),
    ('Cucumber', 'Kano', 'Kano', 102, 62, 98, 6.2, 7.2, 'flowering', 'May-June'),
    ('Cucumber', 'Kano', 'Kano', 82, 48, 78, 6.2, 7.2, 'fruit development', 'May-June'),
    
    ('Cucumber', 'Abuja', 'FCT', 89, 53, 83, 6.1, 7.1, 'vegetative', 'April-May'),
    ('Cucumber', 'Abuja', 'FCT', 103, 63, 99, 6.1, 7.1, 'flowering', 'April-May'),
    ('Cucumber', 'Abuja', 'FCT', 83, 49, 79, 6.1, 7.1, 'fruit development', 'April-May'),
    
    # ==========================================
    # LETTUCE Requirements (Cooler Regions)
    # ==========================================
    ('Lettuce', 'Jos', 'Plateau', 100, 50, 80, 6.0, 7.0, 'vegetative', 'September-November'),
    ('Lettuce', 'Jos', 'Plateau', 115, 60, 95, 6.0, 7.0, 'head formation', 'September-November'),
    
    ('Lettuce', 'Kano', 'Kano', 98, 48, 78, 6.2, 7.2, 'vegetative', 'October-December'),
    ('Lettuce', 'Kano', 'Kano', 112, 58, 92, 6.2, 7.2, 'head formation', 'October-December'),
    
    ('Lettuce', 'Abuja', 'FCT', 99, 49, 79, 6.1, 7.1, 'vegetative', 'September-November'),
    ('Lettuce', 'Abuja', 'FCT', 113, 59, 93, 6.1, 7.1, 'head formation', 'September-November'),
    
    # ==========================================
    # WATERMELON Requirements (All Regions)
    # ==========================================
    ('Watermelon', 'Lagos', 'Lagos', 80, 60, 90, 6.0, 7.0, 'vegetative', 'March-May'),
    ('Watermelon', 'Lagos', 'Lagos', 95, 70, 105, 6.0, 7.0, 'flowering', 'March-May'),
    ('Watermelon', 'Lagos', 'Lagos', 75, 55, 85, 6.0, 7.0, 'fruit development', 'March-May'),
    
    ('Watermelon', 'Kano', 'Kano', 78, 58, 88, 6.2, 7.2, 'vegetative', 'May-June'),
    ('Watermelon', 'Kano', 'Kano', 92, 68, 102, 6.2, 7.2, 'flowering', 'May-June'),
    ('Watermelon', 'Kano', 'Kano', 72, 52, 82, 6.2, 7.2, 'fruit development', 'May-June'),
    
    ('Watermelon', 'Abuja', 'FCT', 79, 59, 89, 6.1, 7.1, 'vegetative', 'April-May'),
    ('Watermelon', 'Abuja', 'FCT', 93, 69, 103, 6.1, 7.1, 'flowering', 'April-May'),
    ('Watermelon', 'Abuja', 'FCT', 73, 53, 83, 6.1, 7.1, 'fruit development', 'April-May'),
    
    # ==========================================
    # PINEAPPLE Requirements (Southern Regions)
    # ==========================================
    ('Pineapple', 'Lagos', 'Lagos', 95, 45, 105, 5.0, 6.0, 'vegetative', 'March-May'),
    ('Pineapple', 'Lagos', 'Lagos', 110, 55, 120, 5.0, 6.0, 'flowering', 'March-May'),
    ('Pineapple', 'Lagos', 'Lagos', 90, 40, 100, 5.0, 6.0, 'fruit development', 'March-May'),
    
    ('Pineapple', 'Benin City', 'Edo', 92, 42, 102, 5.2, 6.2, 'vegetative', 'March-May'),
    ('Pineapple', 'Benin City', 'Edo', 108, 52, 118, 5.2, 6.2, 'flowering', 'March-May'),
    ('Pineapple', 'Benin City', 'Edo', 88, 38, 98, 5.2, 6.2, 'fruit development', 'March-May'),
    
    ('Pineapple', 'Enugu', 'Enugu', 93, 43, 103, 5.1, 6.1, 'vegetative', 'March-May'),
    ('Pineapple', 'Enugu', 'Enugu', 109, 53, 119, 5.1, 6.1, 'flowering', 'March-May'),
    ('Pineapple', 'Enugu', 'Enugu', 89, 39, 99, 5.1, 6.1, 'fruit development', 'March-May'),
    
    # ==========================================
    # MANGO Requirements (All Regions)
    # ==========================================
    ('Mango', 'Kano', 'Kano', 80, 50, 90, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Mango', 'Kano', 'Kano', 95, 60, 105, 5.5, 6.5, 'flowering', 'March-May'),
    ('Mango', 'Kano', 'Kano', 75, 45, 85, 5.5, 6.5, 'fruit development', 'March-May'),
    
    ('Mango', 'Kaduna', 'Kaduna', 78, 48, 88, 5.8, 6.8, 'vegetative', 'March-May'),
    ('Mango', 'Kaduna', 'Kaduna', 92, 58, 102, 5.8, 6.8, 'flowering', 'March-May'),
    ('Mango', 'Kaduna', 'Kaduna', 72, 42, 82, 5.8, 6.8, 'fruit development', 'March-May'),
    
    ('Mango', 'Abuja', 'FCT', 79, 49, 89, 5.7, 6.7, 'vegetative', 'March-May'),
    ('Mango', 'Abuja', 'FCT', 93, 59, 103, 5.7, 6.7, 'flowering', 'March-May'),
    ('Mango', 'Abuja', 'FCT', 73, 43, 83, 5.7, 6.7, 'fruit development', 'March-May'),
    
    # ==========================================
    # ORANGE Requirements (All Regions)
    # ==========================================
    ('Orange', 'Ibadan', 'Oyo', 105, 55, 95, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Orange', 'Ibadan', 'Oyo', 120, 65, 110, 5.5, 6.5, 'flowering', 'March-May'),
    ('Orange', 'Ibadan', 'Oyo', 100, 50, 90, 5.5, 6.5, 'fruit development', 'March-May'),
    
    ('Orange', 'Akure', 'Ondo', 102, 52, 92, 5.5, 6.5, 'vegetative', 'March-May'),
    ('Orange', 'Akure', 'Ondo', 118, 62, 108, 5.5, 6.5, 'flowering', 'March-May'),
    ('Orange', 'Akure', 'Ondo', 98, 48, 88, 5.5, 6.5, 'fruit development', 'March-May'),
    
    ('Orange', 'Jos', 'Plateau', 103, 53, 93, 5.8, 6.8, 'vegetative', 'March-May'),
    ('Orange', 'Jos', 'Plateau', 119, 63, 109, 5.8, 6.8, 'flowering', 'March-May'),
    ('Orange', 'Jos', 'Plateau', 99, 49, 89, 5.8, 6.8, 'fruit development', 'March-May'),
    
    ('Orange', 'Kaduna', 'Kaduna', 104, 54, 94, 5.9, 6.9, 'vegetative', 'March-May'),
    ('Orange', 'Kaduna', 'Kaduna', 120, 64, 110, 5.9, 6.9, 'flowering', 'March-May'),
    ('Orange', 'Kaduna', 'Kaduna', 100, 50, 90, 5.9, 6.9, 'fruit development', 'March-May')
]   
    for req in default_crop_requirements:
        conn.execute('''INSERT OR IGNORE INTO crop_nutrient_requirements 
            (crop_type, region, state, n_requirement, p_requirement, k_requirement, 
             optimal_ph_min, optimal_ph_max, growth_stage, planting_season) 
            VALUES (?,?,?,?,?,?,?,?,?,?)''', req)

    # ==========================================
    # ✅ COMMIT AND CLOSE (MUST BE AT THE VERY END)
    # ==========================================
    conn.commit()
    conn.close()

# Call the function
init_db()

# --- Frontend Routes ---
@app.route('/')
def serve_frontend():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/Register.html')
def serve_register():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'Register.html')

@app.route('/login.html')
def serve_login():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'login.html')

@app.route('/data-acquisition.html')
def serve_data_acquisition():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'data-acquisition.html')

@app.route('/admin.html')
def serve_admin():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'admin.html')

# --- Auth Routes ---
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    full_name = data.get('name', '').strip()
    
    if not email or not password or not full_name:
        return jsonify({"error": "All fields required"}), 400

    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)',
                     (email, password_hash, full_name))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Email exists"}), 409
    finally:
        conn.close()

    return jsonify({
        "message": "Success",
        "access_token": create_access_token(identity=email),
        "user": {"email": email, "name": full_name}
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        # Check if user exists and password is correct
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({"error": "Invalid credentials"}), 401
        
        # ✅ CREATE THE TOKEN (this was missing!)
        token = create_access_token(identity=email)
        
        # ✅ Use correct column names from your database
        # Your users table has: id, email, password_hash, full_name, role
        return jsonify({
            "access_token": token,
            "user": {
                "id": user['id'],
                "email": user['email'],
                "full_name": user['full_name'],
                "role": user['role'],
                "is_admin": user['role'] == 'admin'  # Calculate from role column
            }
        }), 200
        
    except Exception as e:
        print(f"🚨 CRITICAL LOGIN ERROR: {str(e)}")
        return jsonify({"error": "Server error. Check terminal."}), 500

    except Exception as e:
        # THIS LINE IS CRITICAL: It prints the real error to your terminal!
        print(f"🚨 CRITICAL LOGIN ERROR: {str(e)}")
        return jsonify({"error": "Server error. Check terminal."}), 500
# --- Helper Functions ---
TRAIN_MEAN = [0.12, -0.03, -0.15, -0.10, 0.05, 0.02, -0.08, 0.18, -0.04, -0.01, 0.12, 0.48]
TRAIN_STD = [0.75, 0.28, 0.85, 0.72, 1.15, 0.58, 0.38, 0.88, 0.48, 0.09, 0.58, 0.95]
MODEL_FEATURE_ORDER = ["soil_moisture", "EC", "N", "P", "K", "soil_temp", "pH", "air_temp", "humidity", "rainfall", "ndvi_proxy", "growth_stage_encoded"]

FEATURE_MAPPING = {
    "soil_moisture": "soil_moisture",
    "nutrient_ec_dS_m": "EC",
    "npk_n_mgkg": "N",
    "npk_p_mgkg": "P",
    "npk_k_mgkg": "K",
    "soil_temp_c": "soil_temp",
    "soil_ph": "pH",
    "air_temp_c": "air_temp",
    "humidity_pct": "humidity",
    "rainfall_forecast_mm": "rainfall",
    "crop_age_days": "growth_stage_encoded",
    "plant_vi_proxy": "ndvi_proxy"
}

def standardize_input(raw_values):
    return [(0.0 if TRAIN_STD[i] == 0 else (val - TRAIN_MEAN[i]) / TRAIN_STD[i])
            for i, val in enumerate(raw_values)]

def calculate_npk_rates(soil_n, soil_p, soil_k, crop_type="tomato", crop_age_days=60, ndvi=0.5):
    crop_recommendations = {
        "tomato": {"vegetative": {"N": 80, "P": 40, "K": 100}, "flowering": {"N": 100, "P": 50, "K": 130}, "fruiting": {"N": 70, "P": 60, "K": 160}},
        "maize": {"vegetative": {"N": 60, "P": 30, "K": 50}, "flowering": {"N": 80, "P": 40, "K": 65}, "fruiting": {"N": 50, "P": 50, "K": 80}},
        "pepper": {"vegetative": {"N": 70, "P": 35, "K": 80}, "flowering": {"N": 90, "P": 45, "K": 100}, "fruiting": {"N": 60, "P": 55, "K": 120}},
        "cassava": {"vegetative": {"N": 50, "P": 25, "K": 65}, "flowering": {"N": 40, "P": 20, "K": 50}, "fruiting": {"N": 30, "P": 15, "K": 40}},
        "rice": {"vegetative": {"N": 60, "P": 30, "K": 50}, "flowering": {"N": 80, "P": 40, "K": 65}, "fruiting": {"N": 50, "P": 50, "K": 80}},
        "default": {"vegetative": {"N": 60, "P": 30, "K": 65}, "flowering": {"N": 80, "P": 40, "K": 85}, "fruiting": {"N": 55, "P": 50, "K": 105}}
    }
    stage = "vegetative" if crop_age_days < 45 else "flowering" if crop_age_days < 75 else "fruiting"
    targets = crop_recommendations.get(crop_type, crop_recommendations["default"])[stage]

    n_adequate, p_adequate, k_adequate = 30, 15, 100
    n_factor = 1.0 if soil_n >= n_adequate else 1.3
    p_factor = 1.0 if soil_p >= p_adequate else 1.3
    k_factor = 1.0 if soil_k >= k_adequate else 1.3

    if soil_n > 60: n_factor = 0.6
    if soil_p > 30: p_factor = 0.6
    if soil_k > 200: k_factor = 0.6

    n_needed = targets["N"] * n_factor
    p_needed = targets["P"] * p_factor
    k_needed = targets["K"] * k_factor

    if ndvi > 0.65:
        n_needed *= 0.7
        p_needed *= 0.7
        k_needed *= 0.7

    urea_needed = n_needed / 0.46
    p2o5_needed = p_needed * 2.29
    dap_needed = p2o5_needed / 0.46
    n_from_dap = dap_needed * 0.18
    urea_needed = max(0, urea_needed - (n_from_dap / 0.46))
    k2o_needed = k_needed * 1.20
    mop_needed = k2o_needed / 0.60

    def get_status(soil_val, threshold):
        if soil_val < threshold * 0.5: return "DEFICIENT"
        elif soil_val > threshold * 2.5: return "EXCESS"
        return "OPTIMAL"

    n_status = get_status(soil_n, n_adequate)
    p_status = get_status(soil_p, p_adequate)
    k_status = get_status(soil_k, k_adequate)

    needs_fertigation = (n_status == "DEFICIENT" or p_status == "DEFICIENT" or k_status == "DEFICIENT")
    if ndvi > 0.65 and soil_n >= n_adequate and soil_p >= p_adequate and soil_k >= k_adequate:
        needs_fertigation = False

    if not needs_fertigation:
        urea_needed = dap_needed = mop_needed = 0
    else:
        urea_needed = min(urea_needed, 200)
        dap_needed = min(dap_needed, 150)
        mop_needed = min(mop_needed, 150)

    split_needed = (n_needed > 90 or p_needed > 45 or k_needed > 120)
    n_supplied = (urea_needed * 0.46) + (dap_needed * 0.18)
    p2o5_supplied = dap_needed * 0.46
    k2o_supplied = mop_needed * 0.60

    gap_n = max(0, round(n_needed - n_supplied, 1))
    gap_p2o5 = max(0, round(p2o5_needed - p2o5_supplied, 1))
    gap_k2o = max(0, round(k2o_needed - k2o_supplied, 1))

    n_met = n_supplied >= n_needed
    p_met = p2o5_supplied >= p2o5_needed
    k_met = k2o_supplied >= k2o_needed

    return {
        "urea_kg_ha": round(urea_needed, 1), "dap_kg_ha": round(dap_needed, 1), "mop_kg_ha": round(mop_needed, 1),
        "total_n_kg_ha": round(n_needed, 1), "total_p2o5_kg_ha": round(p2o5_needed, 1), "total_k2o_kg_ha": round(k2o_needed, 1),
        "n_supplied_kg_ha": round(n_supplied, 1), "p2o5_supplied_kg_ha": round(p2o5_supplied, 1), "k2o_supplied_kg_ha": round(k2o_supplied, 1),
        "gap_n_kg_ha": gap_n, "gap_p2o5_kg_ha": gap_p2o5, "gap_k2o_kg_ha": gap_k2o,
        "recommendation_status": {
            "N": "SATISFIED" if n_met else "INSUFFICIENT",
            "P": "SATISFIED" if p_met else "INSUFFICIENT",
            "K": "SATISFIED" if k_met else "INSUFFICIENT"
        },
        "needs_fertigation": needs_fertigation, "split_application_needed": split_needed,
        "has_excess": (soil_n > 80 or soil_p > 40 or soil_k > 250),
        "deficiency_details": {
            "n_deficit_kg_ha": round(max(0, targets["N"] - soil_n), 1),
            "p_deficit_kg_ha": round(max(0, targets["P"] - soil_p), 1),
            "k_deficit_kg_ha": round(max(0, targets["K"] - soil_k), 1),
            "status": {"N": n_status, "P": p_status, "K": k_status}
        }
    }

def generate_comprehensive_recommendations(soil_moisture, ec, soil_ph, soil_temp, air_temp, humidity, rainfall, crop_age, ndvi, crop_type, npk_breakdown, soil_n=0, soil_p=0, soil_k=0):
    recommendations = {'irrigation': {}, 'ph_management': {}, 'salinity_management': {}, 'crop_health': {}, 'timing': {}, 'warnings': [], 'actions': []}
    
    if soil_moisture < 25:
        recommendations['irrigation'] = {'needed': True, 'urgency': 'HIGH', 'volume_liters_ha': 40000, 'message': 'Critically low. Irrigate immediately.', 'timing': 'Within 24h'}
        recommendations['actions'].append('Irrigate 40,000 L/ha immediately')
    elif soil_moisture < 30:
        recommendations['irrigation'] = {'needed': True, 'urgency': 'MEDIUM', 'volume_liters_ha': 25000, 'message': 'Below optimal. Schedule soon.', 'timing': 'Within 48h'}
        recommendations['actions'].append('Schedule 25,000 L/ha irrigation')
    elif soil_moisture > 50:
        recommendations['irrigation'] = {'needed': False, 'urgency': 'NONE', 'volume_liters_ha': 0, 'message': 'Adequate. Avoid over-irrigation.'}
        recommendations['warnings'].append('High moisture. Delay irrigation.')
    else:
        recommendations['irrigation'] = {'needed': False, 'urgency': 'NONE', 'volume_liters_ha': 0, 'message': 'Optimal. No irrigation needed.'}

    if rainfall > 20:
        recommendations['irrigation']['needed'] = False
        recommendations['warnings'].append(f'Rainfall expected ({rainfall}mm). Skip irrigation & fertilizer.')

    if soil_ph < 5.5:
        lime = (5.5 - soil_ph) * 2000
        recommendations['ph_management'] = {'needed': True, 'status': 'TOO_ACIDIC', 'action': 'Apply lime', 'amount_kg_ha': round(lime, 0), 'message': f'Acidic (pH {soil_ph}). Apply {lime:.0f} kg/ha lime.'}
        recommendations['actions'].append(f'Apply {lime:.0f} kg/ha lime')
    elif soil_ph > 7.5:
        sulfur = (soil_ph - 7.5) * 500
        recommendations['ph_management'] = {'needed': True, 'status': 'TOO_ALKALINE', 'action': 'Apply sulfur', 'amount_kg_ha': round(sulfur, 0), 'message': f'Alkaline (pH {soil_ph}). Apply {sulfur:.0f} kg/ha sulfur.'}
        recommendations['actions'].append(f'Apply {sulfur:.0f} kg/ha sulfur')
    else:
        recommendations['ph_management'] = {'needed': False, 'status': 'OPTIMAL', 'message': f'Optimal (pH {soil_ph}).'}

    if ec > 4:
        recommendations['salinity_management'] = {'needed': True, 'status': 'HIGH_SALINITY', 'message': f'High EC ({ec}). Leach soil.', 'leaching_volume_liters_ha': 60000}
        recommendations['warnings'].append('High salinity. Leach with 60,000 L/ha.')
    else:
        recommendations['salinity_management'] = {'needed': False, 'status': 'OPTIMAL', 'message': f'Safe EC ({ec}).'}

    if ndvi < 0.3:
        recommendations['crop_health'] = {
            'status': 'POOR',
            'message': f'NDVI ({ndvi}) indicates severe crop stress.',
            'warning': '⚠️ CRITICAL: Soil nutrients may be adequate but crop health is poor. DO NOT apply fertilizer. Investigate immediately.',
            'actions': ['Scout field for pests and diseases', 'Check soil moisture and drainage', 'Inspect root system']
        }
        recommendations['warnings'].append('Poor crop health despite adequate nutrients. Investigate non-nutrient stress factors.')
    elif ndvi < 0.5:
        recommendations['crop_health'] = {'status': 'FAIR', 'message': f'NDVI ({ndvi}) shows moderate stress. Monitor closely.'}
    elif ndvi < 0.7:
        recommendations['crop_health'] = {'status': 'GOOD', 'message': f'NDVI ({ndvi}) indicates healthy crop.'}
    else:
        recommendations['crop_health'] = {'status': 'EXCELLENT', 'message': f'NDVI ({ndvi}) shows excellent health. Minimal inputs needed.'}

    crop_stage = 'vegetative' if crop_age < 45 else 'flowering' if crop_age < 75 else 'fruiting'
    recommendations['timing'] = {'crop_stage': crop_stage, 'optimal_time': 'Early morning or late evening'}

    if air_temp > 35: recommendations['warnings'].append(f'High temp ({air_temp}°C). Avoid fertilizer during heat.')
    if humidity > 85: recommendations['warnings'].append(f'High humidity ({humidity}%). Disease risk.')
    if soil_temp < 18: recommendations['warnings'].append(f'Low soil temp ({soil_temp}°C). Reduced uptake.')
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON payload provided. Please ensure Content-Type is application/json"}), 400

        defaults = {
            "soil_moisture": 35.0, "soil_temp_c": 28.0, "soil_ph": 6.5, "nutrient_ec_dS_m": 1.2,
            "npk_n_mgkg": 25.0, "npk_p_mgkg": 18.0, "npk_k_mgkg": 180.0, "air_temp_c": 30.0,
            "humidity_pct": 65.0, "rainfall_forecast_mm": 0.0, "crop_age_days": 60.0, "plant_vi_proxy": 0.65
        }

        model_payload = {}
        for app_key, model_key in FEATURE_MAPPING.items():
            val = payload.get(app_key)
            if val is None or val == '':
                model_payload[model_key] = defaults.get(app_key, 0.0)
            else:
                model_payload[model_key] = float(val)

        if ML_LOADED:
            raw_values = [model_payload.get(feat, 0) for feat in MODEL_FEATURE_ORDER]
            std_values = standardize_input(raw_values)
            std_payload = dict(zip(MODEL_FEATURE_ORDER, std_values))
            result = predict_with_experts(std_payload, need_threshold=0.45)
        else:
            result = predict_with_experts(model_payload, need_threshold=0.45)

        if 'expert' not in result or 'base' not in result['expert']:
            result['expert'] = {'base': {}}
        base = result['expert']['base']
        base['ts_pred_soil_moisture'] = model_payload.get('soil_moisture', 0)
        base['base_rate_raw'] = result.get('rate_pred', 0)
        base['base_need_proba'] = result.get('need_proba', 0)

        npk_rates = calculate_npk_rates(
            soil_n=model_payload.get('N', 0), soil_p=model_payload.get('P', 0), soil_k=model_payload.get('K', 0),
            crop_type=payload.get('crop_type', 'tomato'), crop_age_days=model_payload.get('growth_stage_encoded', 60),
            ndvi=model_payload.get('ndvi_proxy', 0.5)
        )
        result['fertilizer_breakdown'] = npk_rates

        comprehensive_rec = generate_comprehensive_recommendations(
            soil_moisture=model_payload.get('soil_moisture', 0), ec=model_payload.get('EC', 0),
            soil_ph=model_payload.get('pH', 0), soil_temp=model_payload.get('soil_temp', 0),
            air_temp=model_payload.get('air_temp', 0), humidity=model_payload.get('humidity', 0),
            rainfall=model_payload.get('rainfall', 0), crop_age=model_payload.get('growth_stage_encoded', 0),
            ndvi=model_payload.get('ndvi_proxy', 0), crop_type=payload.get('crop_type', 'tomato'),
            npk_breakdown=npk_rates, soil_n=model_payload.get('N', 0), soil_p=model_payload.get('P', 0), soil_k=model_payload.get('K', 0)
        )
        result['comprehensive_recommendation'] = comprehensive_rec

        irrigation_needed = comprehensive_rec.get('irrigation', {}).get('needed', False)
        fertigation_needed = npk_rates.get('needs_fertigation', False)
        ph_needed = comprehensive_rec.get('ph_management', {}).get('needed', False)
        needs_action = irrigation_needed or fertigation_needed or ph_needed
        
        result['need_label'] = 1 if needs_action else 0
        result['recommendation_text'] = "Action Required" if needs_action else "No Immediate Action Needed"

        if model_payload.get('ndvi_proxy', 0) > 0.8 and not needs_action:
            result['rate_pred'] = 0
            result['need_proba'] = 0.05
            result['need_label'] = 0
            result['timing'] = "N/A"
            if 'expert' in result and 'base' in result['expert']:
                result['expert']['base']['base_rate_raw'] = 0
                result['expert']['base']['base_need_proba'] = 0.05

        try:
            user_email = 'anonymous'
            try:
                verify_jwt_in_request()
                user_email = get_jwt_identity()
            except:
                pass
            
            conn = sqlite3.connect(DB_PATH)
            conn.execute('INSERT INTO prediction_logs (user_email, crop_type) VALUES (?, ?)', 
                        (user_email, payload.get('crop_type', 'unknown')))
            conn.commit()
            conn.close()
        except Exception as log_err:
            print(f"Warning: Could not log prediction: {log_err}")

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ============================================
# ✅ WEATHER & SENSOR API ROUTES
# ============================================
OWM_API_KEY = "0c8a6aa750f31193690c7d7ae248972d"

@app.route('/api/get-weather', methods=['GET'])
def get_weather_proxy():
    city = request.args.get('city', 'Lagos').strip()
    country = request.args.get('country', 'NG').strip()
    
    if not city:
        return jsonify({"error": "City is required"}), 400
        
    try:
        url_current = "http://api.openweathermap.org/data/2.5/weather"
        params_current = {"q": f"{city},{country}", "appid": OWM_API_KEY, "units": "metric"}
        res_current = requests.get(url_current, params=params_current, timeout=10)
        res_current.raise_for_status()
        data_current = res_current.json()
        
        air_temp_c = round(data_current['main']['temp'], 1)
        humidity_pct = data_current['main']['humidity']
        
        url_forecast = "http://api.openweathermap.org/data/2.5/forecast"
        params_forecast = {"q": f"{city},{country}", "appid": OWM_API_KEY, "units": "metric"}
        res_forecast = requests.get(url_forecast, params=params_forecast, timeout=10)
        res_forecast.raise_for_status()
        data_forecast = res_forecast.json()
        
        total_rainfall = 0.0
        for item in data_forecast.get('list', [])[:16]:
            total_rainfall += item.get('rain', {}).get('3h', 0)
            
        return jsonify({
            "success": True,
            "city": data_current.get('name', city),
            "air_temp_c": air_temp_c,
            "humidity_pct": humidity_pct,
            "rainfall_forecast_mm": round(total_rainfall, 1)
        })
        
    except requests.exceptions.HTTPError:
        return jsonify({"success": False, "error": f"City '{city}' not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sensor-ingest', methods=['POST'])
def ingest_sensor_data():
    try:
        data = request.json
        if not data or 'device_id' not in data:
            return jsonify({"error": "device_id is required"}), 400
        
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''INSERT INTO sensor_readings 
            (device_id, soil_moisture, soil_temp_c, soil_ph, nutrient_ec_dS_m, 
             npk_n_mgkg, npk_p_mgkg, npk_k_mgkg, air_temp_c, humidity_pct, 
             rainfall_forecast_mm, crop_age_days, plant_vi_proxy, crop_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (data.get('device_id'), data.get('soil_moisture'), data.get('soil_temp_c'),
             data.get('soil_ph'), data.get('nutrient_ec_dS_m'), data.get('npk_n_mgkg'),
             data.get('npk_p_mgkg'), data.get('npk_k_mgkg'), data.get('air_temp_c'),
             data.get('humidity_pct'), data.get('rainfall_forecast_mm'),
             data.get('crop_age_days'), data.get('plant_vi_proxy'), data.get('crop_type')))
        conn.commit()
        conn.close()
        return jsonify({"message": "Sensor data ingested successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-sensor-data', methods=['GET'])
def get_latest_sensor_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        reading = conn.execute('SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1').fetchone()
        conn.close()
        if reading:
            return jsonify({"success": True, "reading": dict(reading)}), 200
        else:
            return jsonify({"success": False, "message": "No sensor data found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# ✅ ADMIN CONTROLS
# ============================================
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            verify_jwt_in_request()
            email = get_jwt_identity()
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            conn.close()
            if not user or user['role'] != 'admin':
                return jsonify({"error": "Admin access required"}), 403
        except Exception:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    conn = sqlite3.connect(DB_PATH)
    total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_feedback = conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
    total_readings = conn.execute('SELECT COUNT(*) FROM sensor_readings').fetchone()[0]
    conn.close()
    return jsonify({"total_users": total_users, "total_feedback": total_feedback, "total_readings": total_readings})

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    users = conn.execute('SELECT id, email, full_name, role FROM users').fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

@app.route('/api/admin/users', methods=['POST'])
@admin_required
def add_new_user():
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        role = data.get('role', 'user')
        
        if not email or not password or not full_name:
            return jsonify({"error": "Email, password, and full name are required"}), 400
        if role not in ['admin', 'user']:
            return jsonify({"error": "Invalid role"}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute('INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)',
                        (email, password_hash, full_name, role))
            conn.commit()
            return jsonify({"message": "User created successfully", "email": email, "role": role}), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "Email already exists"}), 409
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "User deleted"})

@app.route('/api/admin/users/<int:user_id>/role', methods=['PUT'])
@admin_required
def update_user_role(user_id):
    data = request.json
    new_role = data.get('role')
    if new_role not in ['admin', 'user']:
        return jsonify({"error": "Invalid role"}), 400
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Role updated"}), 200

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user_details(user_id):
    data = request.json
    full_name = data.get('full_name')
    role = data.get('role')
    
    if role not in ['admin', 'user']:
        return jsonify({"error": "Invalid role"}), 400
    if not full_name:
        return jsonify({"error": "Name is required"}), 400

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('UPDATE users SET full_name = ?, role = ? WHERE id = ?', 
                     (full_name, role, user_id))
        conn.commit()
        return jsonify({"message": "User updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/users/<int:user_id>/reset-password', methods=['POST'])
@admin_required
def reset_user_password(user_id):
    new_password = request.json.get('password', 'admin123')
    hashed = generate_password_hash(new_password)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (hashed, user_id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Password reset successful"}), 200

@app.route('/api/admin/settings', methods=['GET'])
@admin_required
def get_settings():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    settings = conn.execute('SELECT * FROM system_settings').fetchall()
    conn.close()
    return jsonify({s['key']: s['value'] for s in settings})

@app.route('/api/admin/settings', methods=['POST'])
@admin_required
def update_settings():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    for key, value in data.items():
        conn.execute('INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)', (key, str(value)))
    conn.commit()
    conn.close()
    return jsonify({"message": "Settings updated"}), 200

@app.route('/api/admin/clear-data', methods=['POST'])
@admin_required
def clear_data():
    table = request.json.get('table')
    if table not in ['feedback', 'sensor_readings']:
        return jsonify({"error": "Invalid table"}), 400
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f'DELETE FROM {table}')
    conn.commit()
    conn.close()
    return jsonify({"message": f"{table} cleared"}), 200

# ============================================
# ✅ CROP, FERTILIZER, REGION, SEASON ROUTES
# ============================================
@app.route('/api/admin/crops', methods=['GET'])
@admin_required
def get_crops():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    crops = conn.execute('SELECT * FROM crops').fetchall()
    conn.close()
    return jsonify([dict(c) for c in crops])

@app.route('/api/admin/crops', methods=['POST'])
@admin_required
def add_crop():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('INSERT INTO crops (name, scientific_name, n_req, p_req, k_req) VALUES (?,?,?,?,?)',
                     (data['name'], data.get('scientific_name', ''), data.get('n_req', 0), data.get('p_req', 0), data.get('k_req', 0)))
        conn.commit()
        return jsonify({"message": "Crop added"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Crop already exists"}), 409
    finally:
        conn.close()

@app.route('/api/admin/crops/<int:crop_id>', methods=['DELETE'])
@admin_required
def delete_crop(crop_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM crops WHERE id = ?', (crop_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Crop deleted"})

# ============================================
# ✅ PUBLIC CROPS ENDPOINT (NEW)
# ============================================
@app.route('/api/crops', methods=['GET'])
def get_public_crops():
    """Public endpoint to get all available crops"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        crops = conn.execute('SELECT name, scientific_name, n_req, p_req, k_req FROM crops ORDER BY name').fetchall()
        conn.close()
        return jsonify([dict(crop) for crop in crops]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route('/api/regions', methods=['GET'])
def get_public_regions():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        regions = conn.execute('SELECT * FROM regional_defaults ORDER BY region_name').fetchall()
        conn.close()
        regions_list = []
        for region in regions:
            regions_list.append({
                'id': region['id'], 'region_name': region['region_name'],
                'default_ph': region['default_ph'], 'default_moisture': region['default_moisture'],
                'typical_n': region['typical_n'], 'typical_p': region['typical_p'],
                'typical_k': region['typical_k'], 'climate_zone': region['climate_zone']
            })
        return jsonify(regions_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/regions', methods=['GET'])
@admin_required
def get_regions():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        regions = conn.execute('SELECT * FROM regional_defaults ORDER BY region_name').fetchall()
        conn.close()
        return jsonify([dict(region) for region in regions]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/regions', methods=['POST'])
@admin_required
def add_region():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''INSERT INTO regional_defaults 
            (region_name, state, default_ph, default_moisture, typical_n, typical_p, typical_k, climate_zone, rainfall_zone) 
            VALUES (?,?,?,?,?,?,?,?,?)''',
            (data['region_name'], data.get('state', ''), data.get('default_ph', 6.5), 
             data.get('default_moisture', 30), data.get('typical_n', 20), 
             data.get('typical_p', 15), data.get('typical_k', 150),
             data.get('climate_zone', 'Savanna'), data.get('rainfall_zone', '')))
        conn.commit()
        conn.close()
        return jsonify({"message": "Region added"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Region already exists"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/regions/<int:region_id>', methods=['DELETE'])
@admin_required
def delete_region(region_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('DELETE FROM regional_defaults WHERE id = ?', (region_id,))
        conn.commit()
        conn.close()
        return jsonify({"message": "Region deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/seasons', methods=['GET'])
def get_public_seasons():
    crop = request.args.get('crop', '')
    region = request.args.get('region', '')
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if crop and region:
        seasons = conn.execute('SELECT * FROM seasonal_calendar WHERE crop_type = ? AND region = ?', (crop, region)).fetchall()
    else:
        seasons = conn.execute('SELECT * FROM seasonal_calendar ORDER BY crop_type, region').fetchall()
    conn.close()
    return jsonify([dict(s) for s in seasons]), 200

@app.route('/api/admin/seasons', methods=['GET'])
@admin_required
def get_seasons():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    seasons = conn.execute('SELECT * FROM seasonal_calendar ORDER BY crop_type, region').fetchall()
    conn.close()
    return jsonify([dict(s) for s in seasons]), 200

@app.route('/api/admin/seasons', methods=['POST'])
@admin_required
def add_season():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''INSERT INTO seasonal_calendar 
            (crop_type, region, state, planting_start, planting_end, harvest_start, harvest_end, 
             rainy_season_start, rainy_season_end, growing_days) 
            VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (data['crop_type'], data['region'], data.get('state', ''), 
             data.get('planting_start', ''), data.get('planting_end', ''),
             data.get('harvest_start', ''), data.get('harvest_end', ''),
             data.get('rainy_season_start', ''), data.get('rainy_season_end', ''),
             data.get('growing_days', 90)))
        conn.commit()
        conn.close()
        return jsonify({"message": "Season added"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Season already exists"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/seasons/<int:season_id>', methods=['DELETE'])
@admin_required
def delete_season(season_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM seasonal_calendar WHERE id = ?', (season_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Season deleted"}), 200

@app.route('/api/admin/fertilizers', methods=['GET'])
@admin_required
def get_fertilizers():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ferts = conn.execute('SELECT * FROM fertilizers').fetchall()
    conn.close()
    return jsonify([dict(f) for f in ferts])

@app.route('/api/admin/fertilizers', methods=['POST'])
@admin_required
def add_fertilizer():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('INSERT INTO fertilizers (name, n_pct, p_pct, k_pct, price_per_kg) VALUES (?,?,?,?,?)',
                     (data['name'], data.get('n_pct', 0), data.get('p_pct', 0), data.get('k_pct', 0), data.get('price', 0)))
        conn.commit()
        return jsonify({"message": "Fertilizer added"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Fertilizer already exists"}), 409
    finally:
        conn.close()

@app.route('/api/admin/fertilizers/<int:fert_id>', methods=['DELETE'])
@admin_required
def delete_fertilizer(fert_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM fertilizers WHERE id = ?', (fert_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Fertilizer deleted"})

# ============================================
# ✅ CROP NUTRIENT REQUIREMENT ROUTES
# ============================================
@app.route('/api/crop-requirements', methods=['GET'])
def get_crop_requirements():
    try:
        crop = request.args.get('crop', '')
        region = request.args.get('region', '')
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        if crop and region:
            requirements = conn.execute('SELECT * FROM crop_nutrient_requirements WHERE crop_type = ? AND region = ? ORDER BY growth_stage', (crop, region)).fetchall()
        elif crop:
            requirements = conn.execute('SELECT * FROM crop_nutrient_requirements WHERE crop_type = ? ORDER BY region, growth_stage', (crop,)).fetchall()
        else:
            requirements = conn.execute('SELECT * FROM crop_nutrient_requirements ORDER BY crop_type, region').fetchall()
        conn.close()
        return jsonify([dict(req) for req in requirements]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/crop-requirements', methods=['GET'])
@admin_required
def get_admin_crop_requirements():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        requirements = conn.execute('SELECT * FROM crop_nutrient_requirements ORDER BY crop_type, region, growth_stage').fetchall()
        conn.close()
        return jsonify([dict(req) for req in requirements]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/crop-requirements', methods=['POST'])
@admin_required
def add_crop_requirement():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''INSERT INTO crop_nutrient_requirements 
            (crop_type, region, state, n_requirement, p_requirement, k_requirement,
             optimal_ph_min, optimal_ph_max, growth_stage, planting_season)
            VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (data['crop_type'], data['region'], data.get('state', ''),
             data.get('n_requirement', 0), data.get('p_requirement', 0), data.get('k_requirement', 0),
             data.get('optimal_ph_min', 5.5), data.get('optimal_ph_max', 7.0),
             data.get('growth_stage', 'vegetative'), data.get('planting_season', '')))
        conn.commit()
        conn.close()
        return jsonify({"message": "Crop requirement added"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Requirement already exists"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/crop-requirements/<int:req_id>', methods=['PUT'])
@admin_required
def update_crop_requirement(req_id):
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''UPDATE crop_nutrient_requirements SET
            n_requirement = ?, p_requirement = ?, k_requirement = ?,
            optimal_ph_min = ?, optimal_ph_max = ?, growth_stage = ?, planting_season = ?
            WHERE id = ?''',
            (data.get('n_requirement', 0), data.get('p_requirement', 0), data.get('k_requirement', 0),
             data.get('optimal_ph_min', 5.5), data.get('optimal_ph_max', 7.0),
             data.get('growth_stage', 'vegetative'), data.get('planting_season', ''), req_id))
        conn.commit()
        conn.close()
        return jsonify({"message": "Requirement updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/crop-requirements/<int:req_id>', methods=['DELETE'])
@admin_required
def delete_crop_requirement(req_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('DELETE FROM crop_nutrient_requirements WHERE id = ?', (req_id,))
        conn.commit()
        conn.close()
        return jsonify({"message": "Requirement deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# ✅ BACKUP & RESTORE ROUTES
# ============================================
@app.route('/api/admin/backup', methods=['GET'])
@admin_required
def create_backup():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'pfdss_backup_{timestamp}.db'
        backup_path = os.path.join(os.path.dirname(DB_PATH), 'backups', backup_filename)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(DB_PATH, backup_path)
        backup_size = os.path.getsize(backup_path)
        backup_size_mb = round(backup_size / (1024 * 1024), 2)
        return jsonify({"success": True, "message": "Backup created", "filename": backup_filename, "size_mb": backup_size_mb}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/backups', methods=['GET'])
@admin_required
def list_backups():
    try:
        backups_dir = os.path.join(os.path.dirname(DB_PATH), 'backups')
        if not os.path.exists(backups_dir):
            return jsonify([]), 200
        backups = []
        for filename in os.listdir(backups_dir):
            if filename.startswith('pfdss_backup_') and filename.endswith('.db'):
                filepath = os.path.join(backups_dir, filename)
                backups.append({
                    "filename": filename,
                    "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
        backups.sort(key=lambda x: x['created'], reverse=True)
        return jsonify(backups), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/restore', methods=['POST'])
@admin_required
def restore_backup():
    try:
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify({"error": "Filename required"}), 400
        backup_path = os.path.join(os.path.dirname(DB_PATH), 'backups', filename)
        if not os.path.exists(backup_path):
            return jsonify({"error": "Backup not found"}), 404
        safety_backup = DB_PATH + '.safety_backup'
        shutil.copy2(DB_PATH, safety_backup)
        shutil.copy2(backup_path, DB_PATH)
        return jsonify({"success": True, "message": "Restored"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/backups/<filename>', methods=['DELETE'])
@admin_required
def delete_backup(filename):
    try:
        backup_path = os.path.join(os.path.dirname(DB_PATH), 'backups', filename)
        if os.path.exists(backup_path):
            os.remove(backup_path)
            return jsonify({"message": "Deleted"}), 200
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# ============================================
# ✅ ANALYTICS ROUTES
# ============================================
@app.route('/api/admin/analytics/overview', methods=['GET'])
@admin_required
def get_analytics_overview():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total_predictions = conn.execute('SELECT COUNT(*) as count FROM prediction_logs').fetchone()['count']
    unique_users = conn.execute('SELECT COUNT(DISTINCT user_email) as count FROM prediction_logs').fetchone()['count']
    today = datetime.now().strftime('%Y-%m-%d')
    predictions_today = conn.execute("SELECT COUNT(*) as count FROM prediction_logs WHERE date(timestamp) = ?", (today,)).fetchone()['count']
    avg_per_user = round(total_predictions / unique_users, 1) if unique_users > 0 else 0
    conn.close()
    return jsonify({"total_predictions": total_predictions, "unique_users": unique_users, "predictions_today": predictions_today, "avg_per_user": avg_per_user}), 200

@app.route('/api/admin/analytics/predictions-over-time', methods=['GET'])
@admin_required
def get_predictions_over_time():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    data = conn.execute('''SELECT date(timestamp) as date, COUNT(*) as count FROM prediction_logs 
        WHERE timestamp >= datetime('now', '-30 days') GROUP BY date(timestamp) ORDER BY date ASC''').fetchall()
    conn.close()
    return jsonify({"labels": [row['date'] for row in data], "values": [row['count'] for row in data]}), 200

@app.route('/api/admin/analytics/crop-distribution', methods=['GET'])
@admin_required
def get_crop_distribution():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    data = conn.execute('SELECT crop_type, COUNT(*) as count FROM prediction_logs GROUP BY crop_type ORDER BY count DESC').fetchall()
    conn.close()
    return jsonify({"labels": [row['crop_type'].title() if row['crop_type'] else 'Unknown' for row in data], "values": [row['count'] for row in data]}), 200

@app.route('/api/admin/analytics/peak-usage', methods=['GET'])
@admin_required
def get_peak_usage():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    data = conn.execute("SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM prediction_logs GROUP BY hour ORDER BY hour ASC").fetchall()
    conn.close()
    hours = list(range(24))
    values = [0] * 24
    for row in data:
        values[int(row['hour'])] = row['count']
    return jsonify({"labels": [f"{h:02d}:00" for h in hours], "values": values}), 200

@app.route('/api/admin/analytics/user-engagement', methods=['GET'])
@admin_required
def get_user_engagement():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    data = conn.execute('''SELECT user_email, COUNT(*) as count, MAX(timestamp) as last_active, MIN(timestamp) as first_active
        FROM prediction_logs GROUP BY user_email ORDER BY count DESC LIMIT 10''').fetchall()
    conn.close()
    users = [{"email": row['user_email'] or 'Anonymous', "predictions": row['count'], "last_active": row['last_active'], "first_active": row['first_active']} for row in data]
    return jsonify({"users": users}), 200

# ============================================
# ✅ IOT & SMART VALVE CONTROL ROUTES
# ============================================
valve_status = {"water_valve": "CLOSED", "fert_valve": "CLOSED", "last_triggered": None, "duration_minutes": 0}

@app.route('/api/iot/calculate-fertigation', methods=['POST'])
def calculate_fertigation():
    try:
        data = request.json
        plot_size_ha = float(data.get('plot_size_ha', 0))
        flow_rate_l_min = float(data.get('flow_rate_l_min', 10))
        injector_ratio = int(data.get('injector_ratio', 100))
        recommendation = data.get('recommendation', {})
        comp_rec = recommendation.get('comprehensive_recommendation', {})
        fb = recommendation.get('fertilizer_breakdown', {})
        
        water_volume_l_ha = comp_rec.get('irrigation', {}).get('volume_liters_ha', 0)
        total_water_liters = water_volume_l_ha * plot_size_ha
        valve_open_minutes = total_water_liters / flow_rate_l_min if flow_rate_l_min > 0 else 0
        
        urea_kg_ha = fb.get('urea_kg_ha', 0)
        dap_kg_ha = fb.get('dap_kg_ha', 0)
        mop_kg_ha = fb.get('mop_kg_ha', 0)
        needs_fertigation = fb.get('needs_fertigation', False)
        
        total_urea_kg = urea_kg_ha * plot_size_ha
        total_dap_kg = dap_kg_ha * plot_size_ha
        total_mop_kg = mop_kg_ha * plot_size_ha
        total_fertilizer_kg = total_urea_kg + total_dap_kg + total_mop_kg
        
        if total_water_liters > 0 and needs_fertigation:
            total_concentration = (total_fertilizer_kg * 1000) / total_water_liters
        else:
            total_concentration = 0
        
        stock_tank_volume_l = total_water_liters / injector_ratio if injector_ratio > 0 else 0
        injection_rate_l_min = flow_rate_l_min / injector_ratio if injector_ratio > 0 else 0
        
        safety_warnings = []
        if total_concentration > 5:
            safety_warnings.append(f"⚠️ High concentration ({total_concentration:.2f} g/L).")
        
        return jsonify({
            "success": True, "plot_size_ha": plot_size_ha, "needs_fertigation": needs_fertigation,
            "water": {"total_liters": round(total_water_liters, 1), "valve_open_minutes": round(valve_open_minutes, 1)},
            "fertilizer_totals": {"urea_kg": round(total_urea_kg, 2), "dap_kg": round(total_dap_kg, 2), "mop_kg": round(total_mop_kg, 2), "total_kg": round(total_fertilizer_kg, 2)},
            "stock_tank": {"volume_liters": round(stock_tank_volume_l, 1), "injector_ratio": f"1:{injector_ratio}", "total_to_dissolve_kg": round(total_fertilizer_kg, 2)},
            "injection": {"rate_l_min": round(injection_rate_l_min, 2), "duration_minutes": round(valve_open_minutes, 1)},
            "safety_warnings": safety_warnings
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/iot/trigger-valves', methods=['POST'])
def trigger_valves():
    try:
        data = request.json
        duration_minutes = float(data.get('duration_minutes', 0))
        needs_fertigation = data.get('needs_fertigation', False)
        if duration_minutes <= 0:
            return jsonify({"error": "Duration must be > 0"}), 400

        valve_status["water_valve"] = "OPEN"
        valve_status["fert_valve"] = "OPEN" if needs_fertigation else "CLOSED"
        valve_status["last_triggered"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        valve_status["duration_minutes"] = duration_minutes
        print(f"🚨 SIMULATION: Valves triggered for {duration_minutes} minutes.")

        return jsonify({"success": True, "message": f"Valves opened for {duration_minutes} minutes.", "status": valve_status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/iot/valve-status', methods=['GET'])
def get_valve_status():
    return jsonify(valve_status), 200

@app.route('/api/iot/stop-valves', methods=['POST'])
def stop_valves():
    valve_status["water_valve"] = "CLOSED"
    valve_status["fert_valve"] = "CLOSED"
    print("🛑 EMERGENCY STOP: Valves closed.")
    return jsonify({
    "success": True, 
    "message": "Valves closed immediately.", 
    "status": valve_status
}), 200

# ============================================
# ✅ SYSTEM HEALTH & AUDIT LOG ROUTES
# ============================================

@app.route('/api/admin/health', methods=['GET'])
@admin_required
def get_system_health():
    """Get system health metrics"""
    try:
        # Get database size
        db_size_bytes = os.path.getsize(DB_PATH)
        db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
        
        # Get server uptime (time since server started)
        start_time = getattr(app, 'start_time', datetime.now())
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        # Get total predictions
        conn = sqlite3.connect(DB_PATH)
        total_predictions = conn.execute('SELECT COUNT(*) FROM prediction_logs').fetchone()[0]
        
        # Get active users (last 24 hours)
        active_users_24h = conn.execute('''
            SELECT COUNT(DISTINCT user_email) FROM prediction_logs 
            WHERE timestamp >= datetime('now', '-24 hours')
        ''').fetchone()[0]
        
        # Get failed logins (last 24 hours) - you'll need to create a login_logs table
        failed_logins_24h = 0
        try:
            failed_logins_24h = conn.execute('''
                SELECT COUNT(*) FROM login_logs 
                WHERE success = 0 AND timestamp >= datetime('now', '-24 hours')
            ''').fetchone()[0]
        except:
            pass  # Table doesn't exist yet
        
        conn.close()
        
        return jsonify({
            "server_status": "Online",
            "db_size_mb": db_size_mb,
            "uptime_seconds": int(uptime_seconds),
            "total_predictions": total_predictions,
            "active_users_24h": active_users_24h,
            "failed_logins_24h": failed_logins_24h
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/audit-logs', methods=['GET'])
@admin_required
def get_audit_logs():
    """Get recent audit logs"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        
        # Try to get from audit_logs table, fallback to prediction_logs
        try:
            logs = conn.execute('''
                SELECT timestamp, user_email, action, details, ip_address 
                FROM audit_logs 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''').fetchall()
        except:
            # Fallback: use prediction_logs as audit trail
            logs = conn.execute('''
                SELECT timestamp, user_email, 
                       'prediction' as action,
                       crop_type as details,
                       'N/A' as ip_address
                FROM prediction_logs 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''').fetchall()
        
        conn.close()
        
        return jsonify([dict(log) for log in logs]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/admin/cleanup-duplicates', methods=['POST'])
@admin_required
def cleanup_duplicates():
    """Remove duplicate seasonal entries"""
    conn = sqlite3.connect(DB_PATH)
    
    # Keep only the first occurrence of each duplicate
    conn.execute('''
        DELETE FROM seasonal_calendar 
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM seasonal_calendar 
            GROUP BY crop_type, region, state
        )
    ''')
    
    deleted_count = conn.total_changes
    conn.commit()
    conn.close()
    
    return jsonify({
        "message": f"Removed {deleted_count} duplicate entries",
        "deleted_count": deleted_count
    }), 200
# --- Start Server ---
if __name__ == '__main__':
    init_db()
    print("🚀 PFDSS Server Running at http://127.0.0.1:5000")
    # Track server start time for uptime calculation
app.start_time = datetime.now()

# --- Start Server ---
if __name__ == '__main__':
    init_db()
    print("🚀 PFDSS Server Running at http://127.0.0.1:5000")
    # ✅ FIXED: Changed host to 127.0.0.1, disabled debug/reloader to prevent Windows socket error
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)