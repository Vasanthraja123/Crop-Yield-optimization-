import math
import random
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, g
import pandas as pd
import os
from functools import wraps
import bcrypt
import uuid
from datetime import datetime, timedelta
import re
import secrets
import logging
from logging.handlers import RotatingFileHandler
import requests
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import json
from dotenv import load_dotenv
from mock_data import (
    get_market_prices_mock,
    get_pest_risk_mock,
    get_soil_data_mock,
    get_weather_data_mock,
    get_mock_dashboard_data,
    recommend_crop
)
from service import init_db
import paho.mqtt.client as mqtt
import threading
import time
import joblib

# Load the crop recommendation model and label encoder once at startup
crop_model = joblib.load('crop_recommender_xgb.pkl')
label_encoder = joblib.load('crop_label_encoder.pkl')

# Load environment variables for API keys
load_dotenv()

# API Keys
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
AMBEE_API_KEY = os.environ.get('AMBEE_API_KEY')
AGROMONITORING_API_KEY = os.environ.get('AGROMONITORING_API_KEY')
COMMODITIES_API_KEY = os.environ.get('COMMODITIES_API_KEY')

# Configuration for IoT sensors
MQTT_BROKER = os.environ.get('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))
MQTT_TOPIC = os.environ.get('MQTT_TOPIC', 'sensors/#')
MQTT_USERNAME = os.environ.get('MQTT_USERNAME', '')
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD', '')

# In-memory storage for sensor data (in production, use a proper database)
sensor_data = {}
sensor_history = {}
sensor_devices = {}

# Toggle to use mock data or real API data
USE_MOCK_DATA = True

# Fallback to mock data if API keys are missing
if not USE_MOCK_DATA:
    missing_keys = []
    if not OPENWEATHER_API_KEY:
        missing_keys.append('OPENWEATHER_API_KEY')
    if not AMBEE_API_KEY:
        missing_keys.append('AMBEE_API_KEY')
    if not AGROMONITORING_API_KEY:
        missing_keys.append('AGROMONITORING_API_KEY')
    if not COMMODITIES_API_KEY:
        missing_keys.append('COMMODITIES_API_KEY')
    if missing_keys:
        logging.Logger.warning(f"Missing API keys: {', '.join(missing_keys)}. Falling back to mock data.")
        USE_MOCK_DATA = True

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=10),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['SESSION_COOKIE_SECURE'] = True  # For HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Define file paths
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

USERS_FILE = os.path.join(DATA_DIR, 'users.xlsx')
CROPS_FILE = os.path.join(DATA_DIR, 'crops.xlsx')
ALERTS_FILE = os.path.join(DATA_DIR, 'alerts.xlsx')

# Initialize data files if they don't exist
def init_data_files():
    # Users file
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=[
            'id', 'username', 'email', 'password_hash', 'full_name',
            'phone', 'region', 'created_at', 'last_login', 'is_active'
        ])
        df.to_excel(USERS_FILE, index=False)

    # Crops file
    if not os.path.exists(CROPS_FILE):
        df = pd.DataFrame(columns=[
            'id', 'user_id', 'crop_name', 'variety', 'planting_date',
            'expected_harvest', 'field_size', 'location', 'notes'
        ])
        df.to_excel(CROPS_FILE, index=False)

    # Alerts file
    if not os.path.exists(ALERTS_FILE):
        df = pd.DataFrame(columns=[
            'id', 'user_id', 'type', 'message', 'created_at', 'is_read'
        ])
        df.to_excel(ALERTS_FILE, index=False)

init_data_files()

# User management functions
def load_users():
    try:
        return pd.read_excel(USERS_FILE)
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return pd.DataFrame(columns=[
            'id', 'username', 'email', 'password_hash', 'full_name',
            'phone', 'region', 'created_at', 'last_login', 'is_active'
        ])

def save_user(username, email, password, full_name, phone="", region=""):
    try:
        df = load_users()

        # Check if username or email already exists
        if username in df['username'].values:
            return False, "Username already exists"

        if email in df['email'].values:
            return False, "Email already exists"

        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Create new user
        new_user = {
            'id': str(uuid.uuid4()),
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'full_name': full_name,
            'phone': phone,
            'region': region,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_login': None,
            'is_active': True
        }

        # Append the new user - this should be updated to use pd.concat
        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True) # Use concat instead for newer pandas versions
        df.to_excel(USERS_FILE, index=False)

        # Create welcome alert for the user
        create_alert(new_user['id'], "welcome", "Welcome to CropMonitor! Start by adding your first crop.")

        return True, "User created successfully"
    except Exception as e:
        logger.error(f"Error saving user: {e}")
        return False, f"Error: {str(e)}"
    
def validate_user(username, password):
    try:
        df = load_users()
        user = df[df['username'] == username]

        if user.empty:
            return False, None

        user = user.iloc[0]
        stored_hash = user['password_hash']

        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            # Update last login
            df.loc[df['username'] == username, 'last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.to_excel(USERS_FILE, index=False)
            return True, user.to_dict()

        return False, None
    except Exception as e:
        logger.error(f"Error validating user: {e}")
        return False, None

def get_user_by_id(user_id):
    df = load_users()
    user = df[df['id'] == user_id]
    if user.empty:
        return None
    return user.iloc[0].to_dict()

def update_user_profile(user_id, full_name=None, email=None, phone=None, region=None):
    try:
        df = load_users()
        user_idx = df.index[df['id'] == user_id].tolist()

        if not user_idx:
            return False, "User not found"

        if full_name:
            df.at[user_idx[0], 'full_name'] = full_name

        if email:
            # Check if email exists for another user
            existing_email = df[(df['email'] == email) & (df['id'] != user_id)]
            if not existing_email.empty:
                return False, "Email already in use"
            df.at[user_idx[0], 'email'] = email

        if phone is not None:
            df.at[user_idx[0], 'phone'] = phone

        if region:
            df.at[user_idx[0], 'region'] = region

        df.to_excel(USERS_FILE, index=False)
        return True, "Profile updated successfully"
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        return False, f"Error: {str(e)}"

def change_password(user_id, current_password, new_password):
    try:
        df = load_users()
        user = df[df['id'] == user_id]

        if user.empty:
            return False, "User not found"

        user = user.iloc[0]

        # Verify current password
        if not bcrypt.checkpw(current_password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return False, "Current password is incorrect"

        # Hash the new password
        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Update password
        df.loc[df['id'] == user_id, 'password_hash'] = new_hash
        df.to_excel(USERS_FILE, index=False)

        return True, "Password changed successfully"
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return False, f"Error: {str(e)}"

# Crop management functions
def load_crops():
    try:
        return pd.read_excel(CROPS_FILE)
    except Exception as e:
        logger.error(f"Error loading crops: {e}")
        return pd.DataFrame(columns=[
            'id', 'user_id', 'crop_name', 'variety', 'planting_date',
            'expected_harvest', 'field_size', 'location', 'notes'
        ])

def get_user_crops(user_id):
    df = load_crops()
    return df[df['user_id'] == user_id].to_dict('records')

def add_crop(user_id, crop_name, variety, planting_date, expected_harvest, field_size, location, notes=""):
    try:
        df = load_crops()

        new_crop = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'crop_name': crop_name,
            'variety': variety,
            'planting_date': planting_date,
            'expected_harvest': expected_harvest,
            'field_size': field_size,
            'location': location,
            'notes': notes
        }

        df = pd.concat([df, pd.DataFrame([new_crop])], ignore_index=True)
        df.to_excel(CROPS_FILE, index=False)

        # Create a new crop alert
        create_alert(user_id, "crop_added", f"New crop added: {crop_name} ({variety})")

        return True, "Crop added successfully"
    except Exception as e:
        logger.error(f"Error adding crop: {e}")
        return False, f"Error: {str(e)}"

def update_crop(crop_id, user_id, **kwargs):
    try:
        df = load_crops()
        crop_idx = df.index[(df['id'] == crop_id) & (df['user_id'] == user_id)].tolist()

        if not crop_idx:
            return False, "Crop not found or unauthorized"

        for key, value in kwargs.items():
            if key in df.columns and key not in ['id', 'user_id']:
                df.at[crop_idx[0], key] = value

        df.to_excel(CROPS_FILE, index=False)
        return True, "Crop updated successfully"
    except Exception as e:
        logger.error(f"Error updating crop: {e}")
        return False, f"Error: {str(e)}"

def delete_crop(crop_id, user_id):
    try:
        df = load_crops()
        before_count = len(df)
        df = df[~((df['id'] == crop_id) & (df['user_id'] == user_id))]

        if len(df) == before_count:
            return False, "Crop not found or unauthorized"

        df.to_excel(CROPS_FILE, index=False)
        return True, "Crop deleted successfully"
    except Exception as e:
        logger.error(f"Error deleting crop: {e}")
        return False, f"Error: {str(e)}"

# Alert management functions
def load_alerts():
    try:
        return pd.read_excel(ALERTS_FILE)
    except Exception as e:
        logger.error(f"Error loading alerts: {e}")
        return pd.DataFrame(columns=[
            'id', 'user_id', 'type', 'message', 'created_at', 'is_read'
        ])

def get_user_alerts(user_id):
    df = load_alerts()
    alerts = df[df['user_id'] == user_id].sort_values('created_at', ascending=False)
    return alerts.to_dict('records')

def create_alert(user_id, alert_type, message):
    try:
        df = load_alerts()

        new_alert = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'type': alert_type,
            'message': message,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_read': False
        }

        df = pd.concat([df, pd.DataFrame([new_alert])], ignore_index=True) # Use concat instead for newer pandas versions
        df.to_excel(ALERTS_FILE, index=False)
        return True
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        return False
    
def mark_alert_read(alert_id, user_id):
    try:
        df = load_alerts()
        alert_idx = df.index[(df['id'] == alert_id) & (df['user_id'] == user_id)].tolist()

        if not alert_idx:
            return False

        df.at[alert_idx[0], 'is_read'] = True
        df.to_excel(ALERTS_FILE, index=False)
        return True
    except Exception as e:
        logger.error(f"Error marking alert as read: {e}")
        return False

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Add this function before the route handlers
def recommend_crop(soil_data, weather_data):
    try:
        # Extract features from soil and weather data (7 features as expected by the model)
        features = np.array([[
            soil_data.get('nitrogen', 50),    # N value
            soil_data.get('phosphorus', 50),  # P value
            soil_data.get('potassium', 50),   # K value
            soil_data.get('ph', 6.5),         # soil pH
            weather_data['main'].get('temp', 25),     # temperature
            weather_data['main'].get('humidity', 60), # humidity
            soil_data.get('moisture', 50),    # soil moisture
        ]])
        
        # Make prediction using the loaded model
        prediction = crop_model.predict(features)
        probabilities = crop_model.predict_proba(features)
        confidence = np.max(probabilities) * 100
        
        # Get the predicted crop name
        crop_name = label_encoder.inverse_transform(prediction)[0]
        
        # Define some sample varieties for each crop
        crop_varieties = {
            'rice': ['IR-8', 'Basmati', 'Jasmine'],
            'wheat': ['HD-2967', 'WH-542', 'PBW-343'],
            'maize': ['DHM-117', 'Vivek QPM-9', 'MAH-14'],
            'chickpea': ['JG-14', 'Vijay', 'KWR-108'],
            'kidneybeans': ['Contender', 'Provider', 'Derby'],
            'pigeonpeas': ['BDN-711', 'BSMR-736', 'ICPL-87119'],
            'mothbeans': ['RMO-40', 'RMO-225', 'RMO-257'],
            'mungbean': ['IPM-02-3', 'SML-668', 'PDM-139'],
            'blackgram': ['TAU-1', 'TPU-4', 'AKU-15'],
            'lentil': ['DPL-62', 'IPL-316', 'JL-3'],
            'pomegranate': ['Bhagwa', 'Ganesh', 'Ruby'],
            'banana': ['Grand Naine', 'Robusta', 'Red Banana'],
            'mango': ['Alphonso', 'Dashehari', 'Langra'],
            'grapes': ['Thompson Seedless', 'Flame Seedless', 'Sharad Seedless'],
            'watermelon': ['Sugar Baby', 'Asahi Yamato', 'Arka Manik'],
            'muskmelon': ['Pusa Sharbati', 'Punjab Sunehri', 'Arka Jeet'],
            'apple': ['Red Delicious', 'Golden Delicious', 'McIntosh'],
            'orange': ['Valencia', 'Nagpur Mandarin', 'Kinnow'],
            'papaya': ['Pusa Delicious', 'Pusa Dwarf', 'Red Lady'],
            'coconut': ['West Coast Tall', 'Chowghat Orange Dwarf', 'Malayan Yellow Dwarf'],
            'cotton': ['Bt Cotton', 'Desi Cotton', 'American Cotton'],
            'jute': ['JRO-524', 'JRO-8432', 'JRO-632'],
            'coffee': ['Arabica', 'Robusta', 'Liberica']
        }
        
        # Get varieties for the predicted crop (default to generic if not found)
        varieties = crop_varieties.get(crop_name.lower(), ['Variety A', 'Variety B', 'Variety C'])
        
        # Generate a realistic yield range based on the crop
        base_yield = random.uniform(2.5, 4.0)  # tons per hectare
        yield_range = f"{base_yield:.1f} - {base_yield + 1.5:.1f} tons/ha"
        
        # Get an appropriate icon (you should have these icons in your static folder)
        icon = f"crops/{crop_name.lower()}.png"
        
        return [{
            'crop': crop_name,
            'variety': random.choice(varieties),
            'confidence': confidence,
            'yield_range': yield_range,
            'icon': icon
        }]
    except Exception as e:
        logger.error(f"Error in recommend_crop: {str(e)}")
        return [{
            'crop': 'Not Available',
            'variety': 'N/A',
            'confidence': 0,
            'yield_range': 'N/A',
            'icon': 'crops/default.png'
        }]

def calculate_crop_health_index(temperature, humidity, rainfall, soil_moisture):
    try:
        # Normalize values to 0-1 range
        temp_score = 1 - abs(temperature - 25) / 25  # Optimal temp around 25°C
        humidity_score = 1 - abs(humidity - 60) / 60  # Optimal humidity around 60%
        rainfall_score = min(rainfall / 50, 1)  # Normalize rainfall (assuming 50mm is optimal)
        moisture_score = 1 - abs(soil_moisture - 50) / 50  # Optimal moisture around 50%
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.3]  # Weights for each factor
        health_index = (
            temp_score * weights[0] +
            humidity_score * weights[1] +
            rainfall_score * weights[2] +
            moisture_score * weights[3]
        ) * 100
        
        return round(health_index, 1)
    except Exception as e:
        logger.error(f"Error calculating crop health index: {str(e)}")
        return 0.0

# Route handlers
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        user_id = session.get('user_id')
        if not user_id:
            flash("Please login first", "warning")
            return redirect(url_for('login'))

        user = get_user_by_id(user_id)
        if not user:
            session.clear()
            flash("User not found", "error")
            return redirect(url_for('login'))

        # Get all dashboard data
        dashboard_data = get_mock_dashboard_data()
        
        # Log the data for debugging
        logger.info(f"Soil Zones Data: {dashboard_data.get('soil_zones')}")
        logger.info(f"Soil Nutrients Data: {dashboard_data.get('soil_nutrients')}")
        logger.info(f"Recommendations Data: {dashboard_data.get('recommendations')}")

        return render_template('dashboard.html',
                            user=user,
                            current_date=datetime.now().strftime('%B %d, %Y'),
                            soil_zones=dashboard_data['soil_zones'],
                            soil_nutrients=dashboard_data['soil_nutrients'],
                            irrigation_tasks=dashboard_data.get('irrigation_tasks', {'tasks': []}),
                            farm_tasks=dashboard_data.get('farm_tasks', []),
                            weather=dashboard_data['weather'],
                            sensor_status=dashboard_data.get('sensor_status', {}),
                            charts=dashboard_data['charts'],
                            market_data=dashboard_data['market_data'],
                            recommendations=dashboard_data.get('recommendations', []))

    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        flash("Error loading dashboard data", "error")
        return redirect(url_for('index'))

@app.route('/crops')
@login_required
def crops():
    user_id = session['user_id']
    crops = get_user_crops(user_id)
    return render_template('crops.html', crops=crops)

@app.route('/crops/add', methods=['GET', 'POST'])
@login_required
def add_crop_route():
    if request.method == 'POST':
        user_id = session['user_id']
        crop_name = request.form['crop_name']
        variety = request.form['variety']
        planting_date = request.form['planting_date']
        expected_harvest = request.form['expected_harvest']
        field_size = request.form['field_size']
        location = request.form['location']
        notes = request.form['notes']

        success, message = add_crop(
            user_id, crop_name, variety, planting_date,
            expected_harvest, field_size, location, notes
        )

        if success:
            flash(message, "success")
            return redirect(url_for('crops'))
        else:
            flash(message, "danger")

    return render_template('add_crop.html')

@app.route('/crops/edit/<crop_id>', methods=['GET', 'POST'])
@login_required
def edit_crop(crop_id):
    user_id = session['user_id']
    df = load_crops()
    crop = df[(df['id'] == crop_id) & (df['user_id'] == user_id)]

    if crop.empty:
        flash("Crop not found or unauthorized", "danger")
        return redirect(url_for('crops'))

    crop = crop.iloc[0].to_dict()

    if request.method == 'POST':
        update_data = {
            'crop_name': request.form['crop_name'],
            'variety': request.form['variety'],
            'planting_date': request.form['planting_date'],
            'expected_harvest': request.form['expected_harvest'],
            'field_size': request.form['field_size'],
            'location': request.form['location'],
            'notes': request.form['notes']
        }

        success, message = update_crop(crop_id, user_id, **update_data)

        if success:
            flash(message, "success")
            return redirect(url_for('crops'))
        else:
            flash(message, "danger")

    return render_template('edit_crop.html', crop=crop)

@app.route('/crops/delete/<crop_id>', methods=['POST'])
@login_required
def delete_crop_route(crop_id):
    user_id = session['user_id']
    success, message = delete_crop(crop_id, user_id)

    if success:
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for('crops'))

@app.route('/alerts')
@login_required
def alerts():
    user_id = session['user_id']
    user_alerts = get_user_alerts(user_id)
    return render_template('alerts.html', alerts=user_alerts)

@app.route('/alerts/mark_read/<alert_id>', methods=['POST'])
@login_required
def mark_alert_read_route(alert_id):
    user_id = session['user_id']
    if mark_alert_read(alert_id, user_id):
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_id = session['user_id']
    user = get_user_by_id(user_id)

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            full_name = request.form.get('fullname')
            email = request.form.get('email')
            phone = request.form.get('phone')
            region = request.form.get('region')

            success, message = update_user_profile(user_id, full_name, email, phone, region)

            if success:
                flash(message, "success")
                return redirect(url_for('settings'))
            else:
                flash(message, "danger")

        elif action == 'change_password':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if new_password != confirm_password:
                flash("New passwords don't match.", "danger")
            else:
                success, message = change_password(user_id, current_password, new_password)
                if success:
                    flash(message, "success")
                    return redirect(url_for('settings'))
                else:
                    flash(message, "danger")

    return render_template('settings.html', user=user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        success, user = validate_user(username, password)

        if success and user:
            session['user_id'] = user['id']
            session.permanent = True
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "danger")

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('fullname')  # This should match the HTML form field name
        phone = request.form.get('phone', '')
        region = request.form.get('region', '')

        # Validate input
        if not username or not email or not password or not full_name:
            flash("All required fields must be filled.", "danger")
        elif password != confirm_password:
            flash("Passwords don't match.", "danger")
        elif not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
            flash("Username must be 3-20 characters and contain only letters, numbers and underscores.", "danger")
        elif not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
            flash("Please enter a valid email address.", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
        else:
            success, message = save_user(username, email, password, full_name, phone, region)

            if success:
                flash(message, "success")
                return redirect(url_for('login'))
            else:
                flash(message, "danger")

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.route('/api/alerts/count')
@login_required
def alert_count():
    user_id = session['user_id']
    user_alerts = get_user_alerts(user_id)
    unread_count = sum(1 for alert in user_alerts if not alert['is_read'])
    return jsonify({'count': unread_count})

@app.route('/api/crops/upcoming_harvests')
@login_required
def upcoming_harvests():
    user_id = session['user_id']
    crops_df = load_crops()
    user_crops = crops_df[crops_df['user_id'] == user_id]

    # Convert expected_harvest to datetime
    user_crops['harvest_date'] = pd.to_datetime(user_crops['expected_harvest'])

    # Get crops with harvest dates in the next 30 days
    today = datetime.now()
    upcoming = user_crops[
        (user_crops['harvest_date'] >= today) &
        (user_crops['harvest_date'] <= today + timedelta(days=30))
    ]

    upcoming = upcoming.sort_values('harvest_date')

    result = []
    for _, crop in upcoming.iterrows():
        days_remaining = (crop['harvest_date'] - today).days
        result.append({
            'id': crop['id'],
            'crop_name': crop['crop_name'],
            'variety': crop['variety'],
            'expected_harvest': crop['expected_harvest'],
            'days_remaining': days_remaining
        })

    return jsonify(result)

@app.route('/api/dashboard-data')
@login_required
def get_dashboard_data():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401

        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404

        dashboard_data = get_mock_dashboard_data()
        
        response_data = {
            'status': 'success',
            'data': {
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'full_name': user['full_name']
                },
                'soil_zones': dashboard_data['soil_zones'],
                'soil_nutrients': dashboard_data['soil_nutrients'],
                'irrigation_tasks': dashboard_data.get('irrigation_tasks', {'tasks': []}),
                'farm_tasks': dashboard_data.get('farm_tasks', []),
                'weather': dashboard_data['weather'],
                'sensor_status': dashboard_data.get('sensor_status', {}),
                'charts': dashboard_data['charts'],
                'market_data': dashboard_data['market_data'],
                'recommendations': dashboard_data.get('recommendations', [])
            }
        }
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in dashboard API: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

# Helper function for security
@app.before_request
def generate_nonce():
    """Generate a unique nonce for each request"""
    g.nonce = secrets.token_urlsafe(16)  # Generate a secure random nonce

@app.context_processor
def inject_nonce():
    return dict(nonce=g.get('nonce', ''))

@app.after_request
def add_security_headers(response):
    nonce = g.get('nonce', '')
    
    # Google Translate is generating dynamically injected inline styles and scripts
    # that we cannot predict or nonce. We need a different approach.
    
    # CSP can be applied in Report-Only mode first to debug
    # Uncomment this line to test in report-only mode first
    # header_name = 'Content-Security-Policy-Report-Only'
    header_name = 'Content-Security-Policy'
    
    # Base sources for scripts
    script_base = [
        "'self'",
        f"'nonce-{nonce}'",
        "https://cdn.jsdelivr.net",
        "https://code.jquery.com",
        "https://stackpath.bootstrapcdn.com",
        "https://translate.google.com",
        "http://translate.google.com",
        "https://translate.googleapis.com",
        "http://translate.googleapis.com",
        "https://translate-pa.googleapis.com",
        "https://cdnjs.cloudflare.com"
    ]
    
    # Style sources
    style_base = [
        "'self'",
        f"'nonce-{nonce}'",
        "https://cdn.jsdelivr.net",
        "https://maxcdn.bootstrapcdn.com",
        "https://cdnjs.cloudflare.com",
        "https://www.gstatic.com",
        "https://fonts.googleapis.com"
    ]
    
    # For Google Translate, we need a special approach:
    # Either add specific hashes for its inline styles and scripts
    # OR use a more permissive policy for development
    
    # List of known hashes from errors
    style_hashes = [
        "'sha256-1mqaE4MG6Bl9KIVVLUhqHKyPnw4Sb3jAw5gRaqRogBU='",
        "'sha256-YcAFp/goa4oZ/go0L/bJqARj1OFlyN88mkdtnxxdwqY='",
        "'sha256-65mkwZPt4V1miqNM9CcVYkrpnlQigG9H6Vi9OM/JCgY='",
        "'sha256-2Ohx/ATsoWMOlFyvs2k+OujvqXKOHaLKZnHMV8PRbIc='"
    ]
    
    script_hashes = [
        "'sha256-RcwiNzBta1hPdiX058KcLE/BH5PoPWU5SJ0+pcIoRZU='"
    ]
    
    # OPTION 1: Development mode - use 'unsafe-inline' without nonces/hashes
    # This is less secure but will allow everything to work
    if app.debug:  # Only use this relaxed policy in development
        csp_parts = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' " + " ".join(script_base[2:]),  # Without nonce
            "script-src-elem 'self' 'unsafe-inline' " + " ".join(script_base[2:]),  # Without nonce
            "style-src 'self' 'unsafe-inline' " + " ".join(style_base[2:]),  # Without nonce
            "style-src-elem 'self' 'unsafe-inline' " + " ".join(style_base[2:]),  # Without nonce
            "img-src 'self' data: https://fonts.gstatic.com https://www.gstatic.com https://png.pngtree.com https://via.placeholder.com https://www.google.com https://translate.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com https://cdnjs.cloudflare.com",
            "connect-src 'self' https://translate.googleapis.com http://translate.googleapis.com https://translate.google.com http://translate.google.com https://translate-pa.googleapis.com"
        ]
    # OPTION 2: Production mode - use nonces and hashes (more secure)
    else:
        # For Google Translate compatibility, we need to have different configurations:
        csp_parts = [
            "default-src 'self'",
            "script-src " + " ".join(script_base + script_hashes),
            "script-src-elem " + " ".join(script_base + script_hashes),
            "style-src " + " ".join(style_base + style_hashes),
            "style-src-elem " + " ".join(style_base + style_hashes),
            "img-src 'self' data: https://fonts.gstatic.com https://www.gstatic.com https://png.pngtree.com https://via.placeholder.com https://www.google.com https://translate.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com https://cdnjs.cloudflare.com",
            "connect-src 'self' https://translate.googleapis.com http://translate.googleapis.com https://translate.google.com http://translate.google.com https://translate-pa.googleapis.com"
        ]
    
    # Join all parts with semicolons
    csp = "; ".join(csp_parts)
    
    # Set the appropriate headers
    response.headers[header_name] = csp
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

if __name__ == '__main__':
    # Initialize required directories
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize data files
    init_data_files()
    
    # Run the application in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)
