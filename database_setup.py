# database_setup.py
import sqlite3

DATABASE_NAME = 'personal_finance.db'

def initialize_db():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create the Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_input TEXT NOT NULL,
            description TEXT,
            amount REAL NOT NULL,
            currency TEXT NOT NULL,
            category TEXT NOT NULL,
            is_income INTEGER NOT NULL,  -- 0 for Expense, 1 for Income
            is_outlier INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_NAME}' initialized successfully.")

if __name__ == '__main__':
    initialize_db()