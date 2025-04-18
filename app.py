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
import numpy as np
import json
from dotenv import load_dotenv
from mock_data import get_market_prices_mock, get_pest_risk_mock, get_soil_data_mock, get_weather_data_mock
from service import init_db


# Load environment variables for API keys
load_dotenv()

# API Keys
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
AMBEE_API_KEY = os.environ.get('AMBEE_API_KEY')
AGROMONITORING_API_KEY = os.environ.get('AGROMONITORING_API_KEY')
COMMODITIES_API_KEY = os.environ.get('COMMODITIES_API_KEY')

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

# Route handlers
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    user = get_user_by_id(user_id)
    crops = get_user_crops(user_id)
    alerts = get_user_alerts(user_id)
    unread_alerts = sum(1 for alert in alerts if not alert['is_read'])
    
    # Get real-time data
    real_time_data = {}
    
    # If there are crops with location data, fetch real-time data for the first one
    if crops and 'location' in crops[0] and crops[0]['location']:
        try:
            lat, lon = map(float, crops[0]['location'].split(','))
            
            # Get weather data
            weather_data = get_weather_data(lat, lon, OPENWEATHER_API_KEY)
            real_time_data['weather'] = weather_data
            
            # Get soil data
            soil_data = get_soil_data(lat, lon, AMBEE_API_KEY)
            real_time_data['soil'] = soil_data
            
            # Get market data for common crops
            commodities = ['WHEAT', 'RICE', 'CORN', 'SOYBEAN']
            market_data = get_market_prices(commodities, COMMODITIES_API_KEY)
            real_time_data['market'] = market_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            flash("Could not fetch all real-time data", "warning")
    
    return render_template(
        'dashboard.html',
        user=user,
        crops=crops,
        recent_alerts=[alert for alert in alerts[:5]],
        unread_alerts=unread_alerts,
        real_time_data=real_time_data,
        current_date=datetime.now().strftime('%B %d, %Y')
    )

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

        if success:
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
    g.nonce = secrets.token_urlsafe(16)

@app.context_processor
def inject_nonce():
    return dict(nonce=g.get('nonce', ''))

# Update the add_security_headers function to better handle Google Translate
@app.after_request
def add_security_headers(response):
    nonce = g.get('nonce', '')
    
    # Define specific style hashes that Google Translate uses
    style_hashes = [
        "'sha256-1mqaE4MG6Bl9KIVVLUhqHKyPnw4Sb3jAw5gRaqRogBU='",
        "'sha256-YcAFp/goa4oZ/go0L/bJqARj1OFlyN88mkdtnxxdwqY='", 
        "'sha256-65mkwZPt4V1miqNM9CcVYkrpnlQigG9H6Vi9OM/JCgY='",
        "'sha256-2Ohx/ATsoWMOlFyvs2k+OujvqXKOHaLKZnHMV8PRbIc='",
         "'nonce-jeG6sngB0s5_COfidshBXw'"
    ]
    
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com https://stackpath.bootstrapcdn.com "
        "http://translate.google.com https://translate.googleapis.com https://translate-pa.googleapis.com https://cdnjs.cloudflare.com; "
        f"style-src 'self' {' '.join(style_hashes)} 'nonce-{nonce}' https://cdn.jsdelivr.net "
        f"https://maxcdn.bootstrapcdn.com https://cdnjs.cloudflare.com https://www.gstatic.com https://fonts.googleapis.com; "
        f"style-src-elem 'self' {' '.join(style_hashes)} 'nonce-{nonce}' https://fonts.googleapis.com "
        f"https://maxcdn.bootstrapcdn.com https://cdnjs.cloudflare.com https://www.gstatic.com; "
        "img-src 'self' data: https://fonts.gstatic.com https://www.gstatic.com https://png.pngtree.com https://via.placeholder.com; "
        "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "connect-src 'self' https://translate.googleapis.com;"
    )
    
    response.headers['Content-Security-Policy'] = csp
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/favicon.ico')
def favicon():
    return '', 204

def get_weather_data(lat, lon, api_key):
    if USE_MOCK_DATA:
        return get_weather_data_mock(lat, lon)
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_soil_data(lat, lon, api_key):
    if USE_MOCK_DATA:
        return get_soil_data_mock(lat, lon)
    url = f"https://api.ambeedata.com/soil/latest/by-lat-lng?lat={lat}&lng={lon}"
    headers = {"x-api-key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def get_pest_risk(lat, lon, crop_type, api_key):
    if USE_MOCK_DATA:
        return get_pest_risk_mock(lat, lon, crop_type)
    url = f"https://api.agromonitoring.com/agro/1.0/pest?lat={lat}&lon={lon}&crop={crop_type}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_market_prices(commodities, api_key):
    if USE_MOCK_DATA:
        return get_market_prices_mock(commodities)
    url = f"https://commodities-api.com/api/latest?access_key={api_key}&base=USD&symbols={','.join(commodities)}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def calculate_crop_health_index(temperature, humidity, rainfall, soil_moisture):
    # Simple model implementation
    # In practice, use a trained model from TensorFlow/PyTorch
    base_score = 50
    
    # Temperature factor (ideal range 20-30°C)
    temp_factor = 100 - (abs(temperature - 25) * 3)
    
    # Moisture factor (ideal range 50-70%)
    moisture_factor = 100 - (abs(soil_moisture - 60) * 2)
    
    # Humidity factor (ideal range 40-70%)
    humidity_factor = 100 - (abs(humidity - 55) * 2)
    
    # Combined health index
    health_index = (temp_factor + moisture_factor + humidity_factor) / 3
    
    # Clamp to 0-100 range
    return max(0, min(100, health_index))


# Train a model first with data like:
# X = np.array([[soil_ph, soil_moisture, temperature, rainfall, ...]])
# y = np.array(['wheat', 'rice', 'maize', ...])

# Save the model
# with open('crop_recommendation_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

def recommend_crop(soil_ph, soil_moisture, temperature, rainfall):
    # Load the model
    with open('crop_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions
    features = np.array([[soil_ph, soil_moisture, temperature, rainfall]])
    prediction = model.predict(features)
    
    # Get probabilities
    probs = model.predict_proba(features)[0]
    
    # Return top 3 recommendations with confidence scores
    top_indices = np.argsort(probs)[-3:][::-1]
    recommendations = []
    
    for idx in top_indices:
        crop = model.classes_[idx]
        confidence = probs[idx] * 100
        recommendations.append({
            'crop': crop,
            'confidence': confidence
        })
    
    return recommendations

@app.route('/api/field-data/')
@login_required
def get_field_data(field_id):
    # Fetch data for specific field using field coordinates
    user_id = session['user_id']
    
    # Get field location from database
    df = load_crops()
    field = df[(df['id'] == field_id) & (df['user_id'] == user_id)]
    
    if field.empty:
        return jsonify({"error": "Field not found"}), 404
    
    # Extract location (assuming format is "lat,lon")
    location = field.iloc[0]['location']
    lat, lon = map(float, location.split(','))
    
    # Get weather data
    weather_data = get_weather_data(lat, lon, OPENWEATHER_API_KEY)
    
    # Get soil data
    soil_data = get_soil_data(lat, lon, AMBEE_API_KEY)
    
    # Get pest risk
    crop_type = field.iloc[0]['crop_name']
    pest_risk = get_pest_risk(lat, lon, crop_type, AGROMONITORING_API_KEY)
    
    # Calculate crop health
    health_index = calculate_crop_health_index(
        weather_data['main']['temp'],
        weather_data['main']['humidity'],
        weather_data.get('rain', {}).get('1h', 0),
        soil_data['soil']['moisture']
    )
    
    return jsonify({
        "field_id": field_id,
        "weather": weather_data,
        "soil": soil_data,
        "pest_risk": pest_risk,
        "crop_health": health_index
    })

from flask import send_file

@app.route('/api/placeholder/<int:width>/<int:height>')
def placeholder_image(width, height):
    # Generate or serve a placeholder image of the specified size
    # For simplicity, serve a local placeholder image or generate dynamically
    # Here, we generate a simple blank image with PIL
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    text = f"{width}x{height}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    textwidth, textheight = draw.textbbox((0, 0), text, font=font)[2:]
    x = (width - textwidth) / 2
    y = (height - textheight) / 2
    draw.text((x, y), text, fill=(50, 50, 50), font=font)

    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    # Call init_db() at the beginning of your app initialization
    init_db()
    init_data_files()  # Your existing function
    app.run(host='0.0.0.0', port=port, debug=debug)
