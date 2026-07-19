"""
============================================================
PFDSS - OpenWeatherMap Integration Script
============================================================
This script fetches weather data from OpenWeatherMap API
and sends it to your PFDSS system.

Author: PFDSS Team
Date: 2026
============================================================
"""

import requests
import json
from datetime import datetime

# ============================================
# ⚙️ CONFIGURATION - EDIT THESE VALUES
# ============================================

# Your PFDSS server URL (change if hosted elsewhere)
PFDSS_API_URL = "http://127.0.0.1:5000/api/sensor-ingest"

# Your OpenWeatherMap API key (replace with your actual key)
OPENWEATHER_API_KEY = "0c8a6aa750f31193690c7d7ae248972d"

# Your location (Nigerian city)
CITY = "Lagos"

# Country code (NG for Nigeria)
COUNTRY_CODE = "NG"

# Device identifier
DEVICE_ID = "weather_api_bot"

# Crop information (optional - update as needed)
CROP_TYPE = "maize"
CROP_AGE_DAYS = 45

# ============================================
# ️ WEATHER DATA FUNCTIONS
# ============================================

def get_current_weather():
    """Fetch current weather data from OpenWeatherMap"""
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{CITY},{COUNTRY_CODE}",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"  # Celsius
    }
    
    try:
        print(f"📡 Fetching weather data for {CITY}, {COUNTRY_CODE}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather_data = {
            "device_id": DEVICE_ID,
            "air_temp_c": round(data['main']['temp'], 1),
            "humidity_pct": data['main']['humidity'],
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get rainfall data if available
        if 'rain' in data:
            rain_1h = data['rain'].get('1h', 0)
            weather_data["rainfall_forecast_mm"] = round(rain_1h, 1)
        else:
            weather_data["rainfall_forecast_mm"] = 0.0
        
        return weather_data
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if response.status_code == 401:
            print("   ⚠️ Your API key is invalid or not activated yet.")
        elif response.status_code == 404:
            print(f"   ⚠️ City '{CITY}' not found.")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - check your internet connection")
        return None
    except requests.exceptions.Timeout:
        print("❌ Request timed out - try again later")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def get_weather_forecast():
    """Fetch 3-hour forecast for rainfall prediction"""
    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": f"{CITY},{COUNTRY_CODE}",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        print(f"📡 Fetching forecast data for {CITY}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Sum rainfall for next 48 hours (16 periods × 3 hours)
        total_rainfall = 0.0
        for item in data['list'][:16]:  # 48 hours
            rain = item.get('rain', {}).get('3h', 0)
            total_rainfall += rain
        
        return round(total_rainfall, 1)
    
    except Exception as e:
        print(f"⚠️ Could not fetch forecast: {e}")
        return 0.0


def send_to_pfdss(data):
    """Send collected data to PFDSS system"""
    try:
        print(f"\n📤 Sending data to PFDSS at {PFDSS_API_URL}...")
        response = requests.post(PFDSS_API_URL, json=data, timeout=10)
        
        if response.status_code == 201:
            print("✅ Data sent successfully to PFDSS!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Failed to send data (HTTP {response.status_code})")
            print(f"   Error: {response.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to PFDSS server!")
        print("   💡 Make sure your PFDSS server is running:")
        print("      python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# ============================================
# 🚀 MAIN EXECUTION
# ============================================

def main():
    print("=" * 60)
    print("🌦️  PFDSS - OpenWeatherMap Data Fetcher")
    print("=" * 60)
    print(f"📍 Location: {CITY}, {COUNTRY_CODE}")
    print(f"🌾 Crop: {CROP_TYPE} ({CROP_AGE_DAYS} days old)")
    print(f"📡 API Key: {OPENWEATHER_API_KEY[:8]}...")
    print("=" * 60)
    print()
    
    # Verify API key is set
    if OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ ERROR: You haven't set your API key yet!")
        print("   📝 Edit fetch_weather.py and replace YOUR_API_KEY_HERE")
        print("      with your actual OpenWeatherMap API key.")
        return
    
    # Step 1: Get current weather
    weather_data = get_current_weather()
    if not weather_data:
        print("\n❌ Failed to get weather data. Exiting.")
        return
    
    print("\n🌡️  Current Weather Data:")
    print(f"   • Temperature: {weather_data['air_temp_c']}°C")
    print(f"   • Humidity: {weather_data['humidity_pct']}%")
    print(f"   • Rainfall (1h): {weather_data['rainfall_forecast_mm']} mm")
    
    # Step 2: Get 48h forecast
    forecast_rainfall = get_weather_forecast()
    weather_data["rainfall_forecast_mm"] = forecast_rainfall
    print(f"   • Rainfall (48h forecast): {forecast_rainfall} mm")
    
    # Step 3: Add crop info
    weather_data["crop_type"] = CROP_TYPE
    weather_data["crop_age_days"] = CROP_AGE_DAYS
    
    # Step 4: Display full payload
    print("\n📊 Full Data Payload:")
    print(json.dumps(weather_data, indent=2))
    
    # Step 5: Send to PFDSS
    print()
    send_to_pfdss(weather_data)
    
    print("\n" + "=" * 60)
    print("✅ Next Steps:")
    print("   1. Open your PFDSS dashboard")
    print("   2. Go to Tab 1 (Data Input)")
    print("   3. Click '🔄 Auto-Fill from External Sensors / API'")
    print("   4. Review the auto-filled values")
    print("   5. Click 'Proceed to Recommendation'")
    print("=" * 60)


if __name__ == "__main__":
    main()