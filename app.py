# app.py (Streamlit UI)
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import time

# Import services and prediction logic
from db_service import add_transaction, get_all_transactions, get_net_balance
from run_app_sim import predict_expense, load_ai_assets # Reuse your powerful AI script!
import plotly.express as px
from datetime import datetime, date
import math

# Color + icon styles for categories (beige & lavender friendly palette)
CATEGORY_STYLES = {
    "Food": {"color": "#BFA5A0", "icon": "🍽️"},
    "Transport": {"color": "#C8B6F6", "icon": "🚌"},
    "Shopping": {"color": "#E6D5FA", "icon": "🛍️"},
    "Salary": {"color": "#D9E6D9", "icon": "💼"},
    "Entertainment": {"color": "#F6D6E8", "icon": "🎮"},
    "Bills": {"color": "#F0E1C8", "icon": "💡"},
    "Health": {"color": "#FBE7C6", "icon": "💊"},
    "Other": {"color": "#E8DFF5", "icon": "🔖"},
}

def get_month_str(dt):
    # Accepts string timestamp or datetime; return YYYY-MM
    if isinstance(dt, str):
        try:
            return dt[:7]
        except Exception:
            return ""
    if isinstance(dt, (datetime, date)):
        return dt.strftime("%Y-%m")
    return ""



# --- Global Initialization ---
# Load assets once at the start
INTERPRETER, CONFIG, CATEGORY_MAP = load_ai_assets()

# --- HELPER FUNCTION: Maps DB results to DataFrame (useful for Streamlit) ---
COLUMN_NAMES = [
    'ID', 'Timestamp', 'Raw Input', 'Description', 'Amount', 
    'Currency', 'Category', 'Is Income', 'Is Outlier'
]
def load_data():
    data = get_all_transactions()
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    df['Is Income'] = df['Is Income'].apply(lambda x: 'Income' if x == 1 else 'Expense')
    df['Amount'] = df['Amount'].apply(lambda x: f"INR {x:.2f}")
    return df

# === UI COMPONENTS ===

def show_overview_and_balance():
    st.header("Home Dashboard 🏠 — At a glance")
    df = load_data()

    if df.empty:
        st.info("No transactions yet. Add your first one from the ➕ tab!")
        return

    # Sidebar filters (month / date range / category / type)
    st.sidebar.subheader("Explore Filters")
    # Month choices (YYYY-MM)
    months = sorted(df['Timestamp'].apply(get_month_str).dropna().unique(), reverse=True)
    sel_month = st.sidebar.selectbox(
    "Month (YYYY-MM)", 
    options=["All"] + months, 
    index=0, 
    key="overview_month"
    )

    start_date = st.sidebar.date_input("From", value=date.today().replace(day=1))
    end_date = st.sidebar.date_input("To", value=date.today())

    categories = ["All"] + sorted(df["Category"].unique().tolist())
    sel_cat = st.sidebar.selectbox("Category", options=categories)
    type_filter = st.sidebar.selectbox("Type", options=["All", "Income", "Expense"])

    # Apply filters
    filtered_df = df.copy()
    # date range filter (timestamps stored as ISO strings or datetimes)
    def in_range(ts):
        try:
            dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        except Exception:
            try:
                dt = datetime.strptime(str(ts), "%Y-%m-%d")
            except:
                return False
        return start_date <= dt.date() <= end_date

    filtered_df = filtered_df[filtered_df['Timestamp'].apply(in_range)]

    if sel_month != "All":
        filtered_df = filtered_df[filtered_df['Timestamp'].str.startswith(sel_month)]

    if sel_cat != "All":
        filtered_df = filtered_df[filtered_df["Category"] == sel_cat]

    if type_filter != "All":
        filtered_df = filtered_df[filtered_df["Is Income"] == type_filter]

    # Ensure Amount numeric
    if filtered_df['Amount'].dtype == 'object':
        filtered_df['Amount_Numeric'] = filtered_df['Amount'].astype(str).str.replace('INR', '', regex=False).str.strip().astype(float)
    else:
        filtered_df['Amount_Numeric'] = filtered_df['Amount']

    total_income = filtered_df[filtered_df['Is Income'] == 'Income']["Amount_Numeric"].sum()
    total_expense = filtered_df[filtered_df['Is Income'] == 'Expense']["Amount_Numeric"].sum()
    net = total_income - total_expense

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Income", f"INR {total_income:.2f}")
    c2.metric("💸 Expense", f"INR {total_expense:.2f}")
    delta_text = f"INR {net:.2f}"
    c3.metric("📊 Net", delta_text, delta=f"INR {net:.2f}")

    st.divider()
    st.subheader("Transactions")
    # Provide ability to view by categories or types via a selectbox
    view_opts = st.selectbox("View transactions grouped by", ["None", "Category", "Type (Income/Expense)"])
    if view_opts == "Category":
        grouped = filtered_df.groupby("Category").agg(count=("ID", "count"), total=("Amount_Numeric", "sum")).reset_index()
        st.table(grouped.sort_values("total", ascending=False))
    elif view_opts == "Type (Income/Expense)":
        grouped = filtered_df.groupby("Is Income").agg(count=("ID", "count"), total=("Amount_Numeric", "sum")).reset_index()
        st.table(grouped)

    st.dataframe(filtered_df.drop(columns=['Amount_Numeric']), use_container_width=True)
    st.caption("Tip: Use the filters on the left to narrow down months, date ranges, category or type.")


def show_input_interface():
    st.header("Add New Transaction")
    # Initialize session state for preview
    if 'preview' not in st.session_state:
        st.session_state['preview'] = None

    # Initialize session state for input storage (separate from widget key)
    if 'raw_input_val' not in st.session_state:
        st.session_state['raw_input_val'] = ""

    # Form for entering raw input
    with st.form("input_form", clear_on_submit=False):
        raw_input = st.text_input(
            "Enter Expense (e.g., 250rs coffee or 50000rs salary)",
            value=st.session_state['raw_input_val'],
            key="raw_input_widget"
        )
        submitted = st.form_submit_button("Enter")  # triggers AI preview

    # Run AI to generate preview when Enter is pressed
    if submitted and raw_input:
        category, amount, is_income, confidence = predict_expense(
            raw_input, INTERPRETER, CONFIG, CATEGORY_MAP
        )
        if amount == 0.0:
            st.error("Failed to extract amount. Use format: [Amount][Currency][Description]")
            st.session_state['preview'] = None
        else:
            st.session_state['preview'] = {
                'raw_input': raw_input,
                'amount': amount,
                'currency': 'INR',
                'category': category,
                'is_income': is_income,
                'confidence': confidence
            }
            # Keep current input in session_state so field remains populated
            st.session_state['raw_input_val'] = raw_input

    # Show AI preview card if available
    if st.session_state['preview']:
        p = st.session_state['preview']
        st.markdown("### Preview")
        st.markdown(f"**Category:** {p['category']}")
        st.markdown(f"**Amount:** INR {p['amount']:.2f}")
        st.markdown(f"**Type:** {'Income' if p['is_income'] else 'Expense'}")
        st.markdown(f"**Raw Input:** {p['raw_input']}")
        st.markdown(f"**Confidence:** {p['confidence']:.2f}")
        st.markdown("#### Press tab twice to come here.")

        # Confirm & Save button
        if st.button("Confirm and Save ✅"):
            new_tx_data = {
                'raw_input': p['raw_input'],
                'description': p['category'],
                'amount': p['amount'],
                'currency': p['currency'],
                'category': p['category'],
                'is_income': p['is_income'],
                'is_outlier': False,
                'timestamp': datetime.now().isoformat()
            }
            add_transaction(new_tx_data)
            st.success("Saved! 🎉")

            # Reset preview and input field for next entry
            st.session_state['preview'] = None
            st.session_state.raw_input = None
            time.sleep(2)
            # Rerun to refresh the form and dashboard
            st.rerun()


import plotly.express as px

def show_reports():
    st.header("Reports & Trends 📊")
    df = load_data()
    if df.empty:
        st.warning("No data to generate reports yet.")
        return

    # Filters in the sidebar for reports
    st.sidebar.subheader("Report Filters")
    months = sorted(df['Timestamp'].apply(get_month_str).dropna().unique(), reverse=True)
    sel_month = st.sidebar.selectbox(
    "Month (YYYY-MM)", 
    options=["All"] + months, 
    index=0, 
    key="reports_month"
)


    # Apply month filter
    working = df.copy()
    if sel_month != "All":
        working = working[working['Timestamp'].str.startswith(sel_month)]

    # Ensure numeric amounts on a working copy
    if working['Amount'].dtype == 'object':
        working['Amount_Numeric'] = working['Amount'].astype(str).str.replace('INR', '', regex=False).str.strip().astype(float)
    else:
        working['Amount_Numeric'] = working['Amount']

    # Expense pie/donut
    expense_df = working[working['Is Income'] == 'Expense'].copy()
    st.subheader("Spending Breakdown")
    if not expense_df.empty:
        cat_sum = expense_df.groupby('Category')['Amount_Numeric'].sum().reset_index()
        fig = px.pie(
            cat_sum,
            names='Category',
            values='Amount_Numeric',
            title=f"Expenses by Category {'('+sel_month+')' if sel_month!='All' else ''}",
            hole=0.35
        )
        fig.update_traces(textinfo='percent+label', pull=[0.04]*len(cat_sum))
        fig.update_layout(transition={'duration': 900, 'easing': 'cubic-in-out'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No expense data for the selected range.")

    # Monthly trend line (last 12 months) for income and expense
    st.subheader("Monthly Trend (Last 12 months)")
    # convert timestamps to month keys
    df_all = df.copy()
    df_all['Month'] = df_all['Timestamp'].apply(get_month_str)
    if df_all['Amount'].dtype == 'object':
        df_all['Amount_Numeric'] = df_all['Amount'].astype(str).str.replace('INR', '', regex=False).str.strip().astype(float)
    else:
        df_all['Amount_Numeric'] = df_all['Amount']

    # prepare pivot
    pivot = (
        df_all.groupby(['Month', 'Is Income'])['Amount_Numeric']
        .sum()
        .reset_index()
        .pivot(index='Month', columns='Is Income', values='Amount_Numeric')
        .fillna(0)
    )
    # pick last 12 months
    pivot = pivot.sort_index().tail(12)
    if pivot.empty:
        st.info("No monthly data available yet.")
    else:
        pivot = pivot.reset_index()
        # Ensure consistent columns
        if 'Income' not in pivot.columns:
            pivot['Income'] = 0
        if 'Expense' not in pivot.columns:
            pivot['Expense'] = 0

        fig2 = px.line(
            pivot,
            x='Month',
            y=['Income', 'Expense'],
            markers=True,
            title="Income vs Expense (monthly)"
        )
        fig2.update_layout(transition={'duration': 700})
        st.plotly_chart(fig2, use_container_width=True)

    # Key summary cards
    total_income = df_all[df_all['Is Income'] == 'Income']['Amount_Numeric'].sum()
    total_expense = df_all[df_all['Is Income'] == 'Expense']['Amount_Numeric'].sum()
    st.metric("Total Income (All time)", f"INR {total_income:.2f}")
    st.metric("Total Expense (All time)", f"INR {total_expense:.2f}")
    st.caption("Charts are interactive — hover and click slices to explore details.")


# app.py (Replace show_data_management function)

from db_service import update_transaction, delete_transaction, add_transaction 
import json # Used for debugging, ensuring compatibility

def show_data_management():
    st.header("Manage & Edit Data 📝")
    
    # Load all data, making ID the key for easy merging and deletion detection
    data_tuples = get_all_transactions()
    column_names = [
        'ID', 'Timestamp', 'Raw Input', 'Description', 'Amount', 
        'Currency', 'Category', 'Is Income', 'Is Outlier'
    ]
    
    # Store original data in session state to compare against edits
    session_key = "original_data"
    if session_key not in st.session_state:
        st.session_state[session_key] = data_tuples
        
    original_df = pd.DataFrame(st.session_state[session_key], columns=column_names)
    
    if original_df.empty:
        st.info("No recorded transactions to manage.")
        return

    # Prepare DataFrame for Display and Editing
    editable_df = original_df[['ID', 'Timestamp', 'Raw Input', 'Amount', 'Currency', 'Is Income', 'Category', 'Is Outlier']].copy()
    editable_df['Is Income'] = editable_df['Is Income'].apply(lambda x: 'Income' if x == 1 else 'Expense')
    
    st.subheader("Edit Transactions")
    
    # Streamlit Data Editor
    edited_df = st.data_editor(
        editable_df, 
        key="data_editor",
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.Column(disabled=True),
            "Timestamp": st.column_config.DatetimeColumn(disabled=True),
            "Is Income": st.column_config.SelectboxColumn(
                width="small",
                options=['Income', 'Expense']
            ),
            "Amount": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True
    )
    
    # --- PROCESSING CHANGES ---

    # 1. Capture the delta (Streamlit's internal method for detecting changes)
    if st.button("Apply Changes and Sync DB 💾"):
        
        try:
            # 1. Check for Deleted Rows
            original_ids = set(original_df['ID'])
            edited_ids = set(edited_df['ID'].dropna()) 
            
            deleted_ids = original_ids - edited_ids
            for tx_id in deleted_ids:
                delete_transaction(tx_id)
                st.info(f"Deleted Transaction ID: {tx_id}")

            # 2. Check for Edited/New Rows
            # Combine the two data frames to iterate over changes efficiently
            
            for index, row in edited_df.iterrows():
                tx_id = row['ID']
                
                # If ID is NaN, it's a new row added by the user in the editor
                is_new_row = pd.isna(tx_id)
                
                # Check for updates only if the ID exists (it's an existing row)
                if not is_new_row:
                    original_row = original_df[original_df['ID'] == tx_id].iloc[0]
                    
                    raw_input_changed = (row['Raw Input'] != original_row['Raw Input'])
                    
                    # Check the RAW INPUT CHANGE requirement
                    if raw_input_changed:
                        st.warning(f"Re-categorizing Row {tx_id} due to Raw Input change: '{row['Raw Input']}'")
                        
                        # RERUN AI PARSING ON NEW RAW INPUT
                        category, amount, is_income, confidence = predict_expense(
                            row['Raw Input'], INTERPRETER, CONFIG, CATEGORY_MAP
                        )
                        
                        # Update the row variables based on AI result
                        row['Amount'] = amount
                        row['Category'] = category
                        # Streamlit data editor returns 'Income'/'Expense' strings, convert back to bool/int
                        row['Is Income'] = True if is_income else False 
                        row['Description'] = category # Use AI category as description
                        # Note: Currency is hardcoded to INR for simplicity

                    # If the row was modified in ANY way (or re-parsed)
                    if not row.equals(original_row) or raw_input_changed:
                        
                        new_data = {
                            'description': row['Description'],
                            'amount': float(row['Amount']),
                            'currency': row['Currency'],
                            'category': row['Category'],
                            'is_income': True if row['Is Income'] == 'Income' else False,
                            'is_outlier': row['Is Outlier'],
                        }
                        update_transaction(tx_id, new_data)
                        st.info(f"Updated Transaction ID: {tx_id}. Category: {new_data['category']}")

                # 3. Handle NEW rows added by the user in the editor
                elif is_new_row and row['Raw Input']:
                    # We treat a new row as a regular ADD operation, and run the AI
                    category, amount, is_income, confidence = predict_expense(
                        row['Raw Input'], INTERPRETER, CONFIG, CATEGORY_MAP
                    )

                    if amount > 0.0:
                        new_tx_data = {
                            'raw_input': row['Raw Input'],
                            'description': category,
                            'amount': amount,
                            'currency': row['Currency'] if row['Currency'] else 'INR',
                            'category': category,
                            'is_income': True if is_income else False,
                            'is_outlier': False
                        }
                        add_transaction(new_tx_data)
                        st.success(f"Added new AI-categorized transaction: {category}")
                        
            
            # Reset session state and rerun to display updates
            del st.session_state[session_key]
            st.success("Database sync complete!")
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during sync: {e}")
            st.json(json.loads(original_df.to_json())) # Debugging output

# NOTE: The helper load_data() function in app.py must be changed to preserve number types:

# app.py (Modified load_data helper function)
def load_data():
    data = get_all_transactions()
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    df['Is Income'] = df['Is Income'].apply(lambda x: 'Income' if x == 1 else 'Expense')
    # FIX: Do NOT format the amount yet, leave it as float for editing
    # df['Amount'] = df['Amount'].apply(lambda x: f"INR {x:.2f}") 
    return df
def main_app():
    st.title("AI Finance Tracker")

    # Use Streamlit Tabs instead of sidebar radio buttons
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview", 
        "➕ Add Transaction", 
        "📋 Reports",
        "✏️ Edit/Manage Data" # New tab for management
    ])

    with tab1:
        show_overview_and_balance()

    with tab2:
        show_input_interface()

    with tab3:
        show_reports()
        
    with tab4:
        show_data_management() # New function to create
        
# Note: You still need to call main_app() at the end of the file.

if __name__ == '__main__':
    # Ensure database is initialized before running the app
    import database_setup
    database_setup.initialize_db() 
    
    main_app()