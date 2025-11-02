# db_service.py
import sqlite3
from typing import List, Tuple, Any

DATABASE_NAME = 'personal_finance.db'

def execute_query(query: str, params: Tuple = ()) -> List[Tuple[Any, ...]]:
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    if query.strip().upper().startswith("SELECT"):
        results = cursor.fetchall()
        conn.close()
        return results
    else:
        conn.commit()
        conn.close()
        return []

def add_transaction(data: dict):
    # data expects: description, amount, currency, category, is_income, is_outlier, raw_input
    query = """
        INSERT INTO transactions (
            raw_input, description, amount, currency, category, is_income, is_outlier
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        data['raw_input'],
        data['description'],
        data['amount'],
        data['currency'],
        data['category'],
        1 if data['is_income'] else 0,
        1 if data['is_outlier'] else 0
    )
    execute_query(query, params)

def get_all_transactions():
    query = "SELECT * FROM transactions ORDER BY timestamp DESC"
    return execute_query(query)

def get_net_balance():
    # Calculate total income and total expense
    income_query = "SELECT SUM(amount) FROM transactions WHERE is_income = 1"
    expense_query = "SELECT SUM(amount) FROM transactions WHERE is_income = 0"
    
    total_income = execute_query(income_query)[0][0] or 0.0
    total_expense = execute_query(expense_query)[0][0] or 0.0
    
    return total_income - total_expense

# db_service.py (Add to the bottom of the file)

def delete_transaction(tx_id: int):
    query = "DELETE FROM transactions WHERE id = ?"
    execute_query(query, (tx_id,))

def update_transaction(tx_id: int, new_data: dict):
    # This is a robust update function that updates multiple fields
    query = """
        UPDATE transactions SET 
            description = ?, amount = ?, currency = ?, 
            category = ?, is_income = ?, is_outlier = ?
        WHERE id = ?
    """
    
    params = (
        new_data['description'],
        new_data['amount'],
        new_data['currency'],
        new_data['category'],
        1 if new_data['is_income'] else 0,
        1 if new_data['is_outlier'] else 0,
        tx_id
    )
    execute_query(query, params)