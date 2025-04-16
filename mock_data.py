# mock_data.py
import math
import random
import json
from datetime import datetime, timedelta

def generate_mock_weather(lat, lon):
    """Generate mock weather data"""
    weather_conditions = [
        {"main": "Clear", "description": "clear sky", "icon": "01d"},
        {"main": "Clouds", "description": "few clouds", "icon": "02d"},
        {"main": "Clouds", "description": "scattered clouds", "icon": "03d"},
        {"main": "Clouds", "description": "broken clouds", "icon": "04d"},
        {"main": "Rain", "description": "light rain", "icon": "10d"},
        {"main": "Thunderstorm", "description": "thunderstorm", "icon": "11d"}
    ]
    
    return {
        "coord": {"lon": lon, "lat": lat},
        "weather": [random.choice(weather_conditions)],
        "main": {
            "temp": round(random.uniform(15, 35), 1),
            "feels_like": round(random.uniform(15, 38), 1),
            "pressure": random.randint(990, 1020),
            "humidity": random.randint(30, 90),
        },
        "wind": {
            "speed": round(random.uniform(1, 20), 1),
            "deg": random.randint(0, 359)
        },
        "clouds": {
            "all": random.randint(0, 100)
        },
        "rain": {
            "1h": round(random.uniform(0, 10), 1) if random.random() < 0.3 else 0
        },
        "dt": int(datetime.now().timestamp()),
        "sys": {
            "sunrise": int((datetime.now().replace(hour=6, minute=0, second=0)).timestamp()),
            "sunset": int((datetime.now().replace(hour=18, minute=0, second=0)).timestamp())
        },
        "name": "Mock Location"
    }

def generate_mock_soil_data(lat, lon):
    """Generate mock soil data"""
    return {
        "soil": {
            "moisture": round(random.uniform(40, 80), 1),
            "temperature": round(random.uniform(18, 28), 1),
            "ph": round(random.uniform(5.5, 7.5), 1),
            "nutrients": {
                "nitrogen": round(random.uniform(30, 80), 1),
                "phosphorus": round(random.uniform(20, 60), 1),
                "potassium": round(random.uniform(25, 70), 1)
            }
        }
    }

def generate_mock_pest_risk(lat, lon, crop_type):
    """Generate mock pest risk data"""
    pests = {
        "wheat": ["aphids", "beetles", "grasshoppers"],
        "rice": ["stem borers", "leaf hoppers", "rice bugs"],
        "corn": ["armyworms", "corn borers", "rootworms"],
        "chickpea": ["pod borers", "aphids", "cutworms"],
        "mustard": ["aphids", "flea beetles", "weevils"]
    }
    
    crop_pests = pests.get(crop_type.lower(), ["aphids", "beetles", "mites"])
    
    risk_level = random.random()  # 0-1 risk score
    risk_text = "Low" if risk_level < 0.33 else "Medium" if risk_level < 0.66 else "High"
    
    return {
        "risk": risk_level,
        "risk_level": risk_text,
        "potential_pests": random.sample(crop_pests, k=min(2, len(crop_pests))),
        "recommendations": [
            "Regular monitoring of field borders",
            "Check for pest signs on underside of leaves",
            "Consider biological control methods"
        ]
    }

def generate_mock_market_prices(commodities):
    """Generate mock market price data"""
    result = {
        "data": {},
        "base": "USD",
        "timestamp": int(datetime.now().timestamp())
    }
    
    for commodity in commodities:
        base_price = {
            "WHEAT": 2150,
            "RICE": 3200,
            "CORN": 1820,
            "CHICKPEA": 4900,
            "MUSTARD": 5280,
            "SOYBEAN": 3600
        }.get(commodity.upper(), 2000)
        
        # Add some randomness
        price = base_price * (1 + random.uniform(-0.05, 0.05))
        
        # Generate price trend (last 5 days)
        trend = []
        current_price = price
        for _ in range(5):
            trend.append(round(current_price, 2))
            current_price = current_price * (1 + random.uniform(-0.02, 0.02))
        
        # Calculate change percentage
        change_pct = ((price - trend[-1]) / trend[-1]) * 100
        
        result["data"][commodity] = {
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "trend": trend
        }
    
    return result

def generate_mock_soil_history(field_id, days=30):
    """Generate mock soil history data"""
    now = datetime.now()
    timestamps = [(now - timedelta(days=i)).timestamp() for i in range(days)]
    
    # Generate realistic soil moisture with gradual changes and occasional spikes after "rain"
    moisture = [random.uniform(55, 65)]
    for i in range(1, days):
        # Normal decay
        new_moisture = max(30, moisture[-1] * 0.98)
        
        # Occasional rain
        if random.random() < 0.2:
            new_moisture = min(85, new_moisture + random.uniform(10, 20))
        
        moisture.append(new_moisture)
    
    # Generate soil temperature
    temperature = []
    for i in range(days):
        # Temperature follows a more gradual trend
        base_temp = 22 + 3 * math.sin(i / 10)  # Gentle sine wave
        temp = base_temp + random.uniform(-1.5, 1.5)
        temperature.append(temp)
    
    return {
        "field_id": field_id,
        "timestamps": sorted(timestamps),  # Sort from oldest to newest
        "moisture": list(reversed(moisture)),  # Align with timestamps
        "temperature": list(reversed(temperature))  # Align with timestamps
    }

# Use these mock functions in your routes
def get_weather_data_mock(lat, lon, api_key=None):
    return generate_mock_weather(lat, lon)

def get_soil_data_mock(lat, lon, api_key=None):
    return generate_mock_soil_data(lat, lon)

def get_pest_risk_mock(lat, lon, crop_type, api_key=None):
    return generate_mock_pest_risk(lat, lon, crop_type)

def get_market_prices_mock(commodities, api_key=None):
    return generate_mock_market_prices(commodities)