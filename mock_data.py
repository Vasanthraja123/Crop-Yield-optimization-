# mock_data.py
import math
import random
import json
from datetime import datetime, timedelta
import numpy as np

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
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate base values
    moisture = round(random.uniform(40, 80), 1)
    temperature = round(random.uniform(18, 28), 1)
    ph = round(random.uniform(5.5, 7.5), 1)
    nitrogen = round(random.uniform(40, 120), 1)
    phosphorus = round(random.uniform(20, 80), 1)
    potassium = round(random.uniform(30, 100), 1)
    
    # Generate soil zones with realistic moisture values
    zones = {
        'A': {
            'status': 'Optimal',
            'moisture': round(random.uniform(60, 75), 1)
        },
        'B': {
            'status': 'Good',
            'moisture': round(random.uniform(45, 59), 1)
        },
        'C': {
            'status': 'Needs Attention',
            'moisture': round(random.uniform(30, 44), 1)
        }
    }
    
    return {
        'moisture': moisture,
        'temperature': temperature,
        'ph': ph,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'zones': zones,
        'last_updated': current_time
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

def recommend_crop(soil_data, weather_data):
    """Generate mock crop recommendations"""
    try:
        # Extract features from soil and weather data
        soil_moisture = soil_data.get('moisture', 60)
        soil_ph = soil_data.get('ph', 6.5)
        temperature = weather_data['main'].get('temp', 25)
        humidity = weather_data['main'].get('humidity', 60)

        # Define crop options based on conditions
        crops = [
            {
                'crop': 'Wheat',
                'variety': 'HD-2967',
                'confidence': 92.5,
                'yield_range': '4.5 - 5.5 tons/ha',
                'icon': 'crops/wheat.png'
            },
            {
                'crop': 'Rice',
                'variety': 'Basmati-370',
                'confidence': 88.0,
                'yield_range': '3.8 - 4.8 tons/ha',
                'icon': 'crops/rice.png'
            },
            {
                'crop': 'Maize',
                'variety': 'DHM-117',
                'confidence': 85.5,
                'yield_range': '5.0 - 6.0 tons/ha',
                'icon': 'crops/maize.png'
            }
        ]

        # Adjust confidence based on conditions
        for crop in crops:
            base_confidence = crop['confidence']
            
            # Adjust for soil moisture
            if 55 <= soil_moisture <= 75:
                base_confidence += 5
            elif soil_moisture < 40 or soil_moisture > 85:
                base_confidence -= 10

            # Adjust for temperature
            if 20 <= temperature <= 30:
                base_confidence += 5
            elif temperature < 15 or temperature > 35:
                base_confidence -= 10

            # Ensure confidence stays within bounds
            crop['confidence'] = max(min(base_confidence, 100), 0)

        # Sort by confidence and return top 2
        sorted_crops = sorted(crops, key=lambda x: x['confidence'], reverse=True)
        return sorted_crops[:2]

    except Exception as e:
        logger.error(f"Error in recommend_crop: {str(e)}")
        return [{
            'crop': 'Not Available',
            'variety': 'N/A',
            'confidence': 0,
            'yield_range': 'N/A',
            'icon': 'crops/default.png'
        }]

# Use these mock functions in your routes
def get_weather_data_mock(lat, lon, api_key=None):
    return generate_mock_weather(lat, lon)

def get_soil_data_mock(lat, lon, api_key=None):
    return generate_mock_soil_data(lat, lon)

def get_pest_risk_mock(lat, lon, crop_type, api_key=None):
    return generate_mock_pest_risk(lat, lon, crop_type)

def get_market_prices_mock(commodities, api_key=None):
    return generate_mock_market_prices(commodities)

def get_mock_soil_zones():
    return {
        'A': {
            'status': 'Optimal',
            'moisture': 75
        },
        'B': {
            'status': 'Good',
            'moisture': 65
        },
        'C': {
            'status': 'Needs Attention',
            'moisture': 45
        }
    }

def get_mock_weather_data():
    return {
        'main': {
            'temp': 25.6,
            'humidity': 65,
            'pressure': 1012
        },
        'weather': [{
            'description': 'Partly cloudy',
            'icon': '02d'
        }],
        'wind': {
            'speed': 3.6,
            'deg': 220
        }
    }

def get_mock_market_data():
    return {
        'table': {
            'crops': [
                {'name': 'Wheat', 'price': 2150, 'change': 2.5},
                {'name': 'Rice', 'price': 1850, 'change': -1.2},
                {'name': 'Corn', 'price': 1650, 'change': 1.8},
                {'name': 'Soybean', 'price': 3200, 'change': 0.5}
            ]
        },
        'chart': {
            'dates': ['Mar 21', 'Mar 28', 'Apr 4', 'Apr 11', 'Apr 18'],
            'values': [2100, 2150, 2120, 2180, 2150]
        }
    }

def get_mock_chart_data():
    timestamps = [(datetime.now() - timedelta(hours=x)).strftime('%H:%M') for x in range(24, -1, -1)]
    return {
        'soil': {
            'timestamps': timestamps,
            'moisture': [random.randint(60, 80) for _ in range(25)]
        },
        'npk': {
            'timestamps': timestamps,
            'nitrogen': [random.randint(70, 90) for _ in range(25)],
            'phosphorus': [random.randint(40, 60) for _ in range(25)],
            'potassium': [random.randint(50, 70) for _ in range(25)]
        }
    }

def generate_mock_irrigation_tasks():
    """Generate mock irrigation tasks"""
    now = datetime.now()
    tasks = []
    
    # Generate 3-5 irrigation tasks
    for i in range(random.randint(3, 5)):
        task_date = now + timedelta(days=random.randint(0, 7))
        task_time = f"{random.randint(6, 18):02d}:00"
        duration = f"{random.randint(30, 120)} minutes"
        
        tasks.append({
            'field': f'Field {random.randint(1, 5)}',
            'date': task_date.strftime('%Y-%m-%d'),
            'time': task_time,
            'duration': duration,
            'priority': random.choice(['high', 'medium', 'low'])
        })
    
    return {'tasks': sorted(tasks, key=lambda x: (x['date'], x['time']))}

def generate_mock_farm_tasks():
    """Generate mock farm tasks"""
    now = datetime.now()
    tasks = []
    
    task_types = [
        ('Fertilizer Application', 'Apply NPK fertilizer'),
        ('Pest Control', 'Spray organic pesticide'),
        ('Soil Testing', 'Collect soil samples for analysis'),
        ('Crop Inspection', 'Check for signs of disease'),
        ('Weed Control', 'Remove weeds manually')
    ]
    
    # Generate 4-6 farm tasks
    for i in range(random.randint(4, 6)):
        task_type, description = random.choice(task_types)
        due_date = now + timedelta(days=random.randint(1, 14))
        
        tasks.append({
            'title': task_type,
            'description': description,
            'field': f'Field {random.randint(1, 5)}',
            'due_date': due_date.strftime('%Y-%m-%d'),
            'priority': random.choice(['high', 'medium', 'low']),
            'status': random.choice(['pending', 'in_progress', 'completed'])
        })
    
    return sorted(tasks, key=lambda x: x['due_date'])

def generate_irrigation_rainfall_data(days=7):
    """Generate mock irrigation and rainfall data for the specified number of days"""
    now = datetime.now()
    data = {
        'dates': [],
        'irrigation': [],
        'rainfall': []
    }
    
    for i in range(days, -1, -1):
        date = now - timedelta(days=i)
        data['dates'].append(date.strftime('%Y-%m-%d'))
        data['irrigation'].append(round(random.uniform(2.0, 8.0), 1))  # 2-8mm irrigation
        data['rainfall'].append(round(random.uniform(0.0, 5.0), 1))    # 0-5mm rainfall
    
    return data

def get_mock_dashboard_data():
    # Get base data
    weather = get_weather_data_mock(0, 0)
    soil_data = get_soil_data_mock(0, 0)
    market_data = get_market_prices_mock(['wheat', 'rice', 'corn'])
    
    # Generate timestamps for the last 24 hours
    now = datetime.now()
    timestamps = [(now - timedelta(hours=x)).strftime('%Y-%m-%d %H:%M:%S') for x in range(24, -1, -1)]
    
    # Get irrigation and rainfall data
    irrigation_data = generate_irrigation_rainfall_data(7)

    # Generate market price data
    market_chart_data = {
        'dates': [(now - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, -1, -1)],
        'prices': {
            'Wheat': [round(random.uniform(2000, 2300), 2) for _ in range(31)],
            'Rice': [round(random.uniform(1800, 2100), 2) for _ in range(31)],
            'Corn': [round(random.uniform(1500, 1800), 2) for _ in range(31)]
        }
    }

    # Current market prices for the table
    current_prices = [
        {'crop': 'Wheat', 'price': market_chart_data['prices']['Wheat'][-1], 
         'change': round(((market_chart_data['prices']['Wheat'][-1] - market_chart_data['prices']['Wheat'][-2]) / market_chart_data['prices']['Wheat'][-2]) * 100, 1)},
        {'crop': 'Rice', 'price': market_chart_data['prices']['Rice'][-1],
         'change': round(((market_chart_data['prices']['Rice'][-1] - market_chart_data['prices']['Rice'][-2]) / market_chart_data['prices']['Rice'][-2]) * 100, 1)},
        {'crop': 'Corn', 'price': market_chart_data['prices']['Corn'][-1],
         'change': round(((market_chart_data['prices']['Corn'][-1] - market_chart_data['prices']['Corn'][-2]) / market_chart_data['prices']['Corn'][-2]) * 100, 1)}
    ]

    # Generate crop recommendations
    soil_conditions = soil_data
    recommendations = recommend_crop(soil_conditions, weather)

    return {
        'weather': weather,
        'soil_zones': get_mock_soil_zones(),
        'soil_nutrients': soil_data,
        'irrigation_data': irrigation_data,
        'market_data': {
            'chart': market_chart_data,
            'current_prices': current_prices
        },
        'charts': {
            'soil': {
                'timestamps': timestamps,
                'moisture': [random.randint(60, 80) for _ in range(25)],
                'temperature': [random.randint(20, 30) for _ in range(25)]
            },
            'npk': {
                'timestamps': timestamps,
                'nitrogen': [random.randint(40, 120) for _ in range(25)],
                'phosphorus': [random.randint(20, 80) for _ in range(25)],
                'potassium': [random.randint(30, 100) for _ in range(25)]
            }
        },
        'irrigation_tasks': generate_mock_irrigation_tasks(),
        'farm_tasks': generate_mock_farm_tasks(),
        'recommendations': recommendations
    }