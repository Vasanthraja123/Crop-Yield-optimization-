import sqlite3

def init_db():
    conn = sqlite3.connect('cropmonitor.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        phone TEXT,
        region TEXT,
        created_at TEXT,
        last_login TEXT,
        is_active INTEGER
    )
    ''')
    
    # Create crops table
    c.execute('''
    CREATE TABLE IF NOT EXISTS crops (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        crop_name TEXT NOT NULL,
        variety TEXT,
        planting_date TEXT,
        expected_harvest TEXT,
        field_size TEXT,
        location TEXT,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create alerts table
    c.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        type TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TEXT,
        is_read INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()