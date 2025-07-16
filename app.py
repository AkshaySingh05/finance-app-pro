# finance_app_pro/app.py

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DB_PATH = "personal_finance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS Transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, type TEXT, category TEXT,
        amount REAL, note TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT, month TEXT, budget_amount REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Debts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        creditor TEXT, balance REAL, interest_rate REAL,
        due_date TEXT, min_payment REAL, credit_limit REAL, initial_balance REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT, target_amount REAL, current_amount REAL, deadline TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, type TEXT)''')
    conn.commit()
    conn.close()

# Data functions (get, add)
def get_categories(type_filter=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT name FROM Categories"
    if type_filter:
        query += f" WHERE type='{type_filter}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df['name'].tolist()

def add_category(name, type_):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO Categories (name, type) VALUES (?, ?)", (name, type_))
    conn.commit()
    conn.close()

def add_transaction(date, type_, category, amount, note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT INTO Transactions (date, type, category, amount, note)
                    VALUES (?, ?, ?, ?, ?)''', (date, type_, category, amount, note))
    conn.commit()
    conn.close()

def get_transactions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM Transactions ORDER BY date DESC", conn)
    conn.close()
    return df

def get_debts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM Debts", conn)
    conn.close()
    return df

def add_debt(creditor, balance, rate, due, min_pay, credit_limit):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO Debts (creditor, balance, interest_rate, due_date, min_payment, credit_limit, initial_balance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (creditor, balance, rate, due, min_pay, credit_limit, balance))
    conn.commit()
    conn.close()

def add_goal(description, target, current, deadline):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO Goals (description, target_amount, current_amount, deadline)
        VALUES (?, ?, ?, ?)
    """, (description, target, current, deadline))
    conn.commit()
    conn.close()

def add_budget(category, month, amount):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO Budgets (category, month, budget_amount) VALUES (?, ?, ?)",
                 (category, month, amount))
    conn.commit()
    conn.close()

def debt_payoff_simulator(debts, method='snowball'):
    if debts.empty:
        return pd.DataFrame()

    # Sort debts: snowball (lowest balance first), avalanche (highest interest rate first)
    if method == 'snowball':
        debts = debts.sort_values('balance')
    else:
        debts = debts.sort_values('interest_rate', ascending=False)

    results = []
    for _, row in debts.iterrows():
        balance = row['balance']
        rate = row['interest_rate'] / 100 / 12  # Monthly interest rate
        payment = row['min_payment']
        months = 0

        if payment <= balance * rate:
            results.append({"Creditor": row['creditor'], "Months to Payoff": "∞ (Insufficient Payment)"})
            continue

        while balance > 0:
            interest = balance * rate
            balance = max(0, balance + interest - payment)
            months += 1

        results.append({"Creditor": row['creditor'], "Months to Payoff": months})

    return pd.DataFrame(results)

def main():
    st.set_page_config("Finance App – Pro Edition", layout="wide")
    init_db()

    menu = st.sidebar.radio("Navigation", ["➕ Add Transaction", "🏋️ KPIs & Dashboard", "📈 Forecasting",
                                       "🌟 Goals", "📆 Debt Simulator", "💰 Budgets", "🌐 Manage Categories"])

    if menu == "➕ Add Transaction":
        st.subheader("Add New Transaction")
        with st.form("txn_form"):
            date = st.date_input("Date")
            type_ = st.selectbox("Type", ["Income", "Expense", "Debt Payment"])
            category = st.selectbox("Category", get_categories())
            amount = st.number_input("Amount", step=0.01)
            note = st.text_input("Note")
            if st.form_submit_button("Submit"):
                add_transaction(str(date), type_, category, amount, note)
                st.success("Transaction added successfully!")


    elif menu == "🏋️ KPIs & Dashboard":
        df = get_transactions()
        df['date'] = pd.to_datetime(df['date'])
        debts = get_debts()
        income = df[df['type'] == 'Income']['amount'].sum()
        expenses = df[df['type'] == 'Expense']['amount'].sum()
        debt_payments = df[df['type'] == 'Debt Payment']['amount'].sum()
        savings = income + expenses + debt_payments
        savings_rate = (savings / income * 100) if income > 0 else 0
        expense_ratio = (-expenses / income * 100) if income > 0 else 0

        if not debts.empty:
            debts['credit_util'] = debts['balance'] / debts['credit_limit'] * 100
            debts['debt_reduction'] = (debts['initial_balance'] - debts['balance']) / debts['initial_balance'] * 100
        credit_util = debts['credit_util'].mean() if not debts.empty else 0
        debt_reduction = debts['debt_reduction'].mean() if not debts.empty else 0
        emergency = df[df['category'] == 'Emergency Fund']['amount'].sum()
        months_exp = -expenses / len(df['date'].dt.to_period("M").unique()) if expenses < 0 else 0
        emergency_months = emergency / months_exp if months_exp > 0 else 0

        st.subheader("📊 Key Financial Indicators")
        st.metric("Savings Rate (%)", f"{savings_rate:.1f}%")
        st.metric("Expense Ratio (%)", f"{expense_ratio:.1f}%")
        st.metric("Net Cash Flow", f"{income + expenses + debt_payments:.2f}")
        st.metric("Avg Credit Utilization (%)", f"{credit_util:.1f}%")
        st.metric("Avg Debt Reduction (%)", f"{debt_reduction:.1f}%")
        st.metric("Emergency Fund Coverage (months)", f"{emergency_months:.1f}%")

    elif menu == "📈 Forecasting":
        st.subheader("Expense Forecast")
        df = get_transactions()
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['type'] == 'Expense']
        df = df.groupby(df['date'].dt.to_period("M")).sum().reset_index()
        df['date'] = df['date'].astype(str)
        df['month_index'] = range(len(df))
        if len(df) >= 2:
            X = df[['month_index']]
            y = df['amount']
            model = LinearRegression().fit(X, y)
            future_idx = np.array([[len(df)+i] for i in range(6)])
            preds = model.predict(future_idx)
            plt.plot(df['date'], y, marker='o', label='Actual')
            plt.plot([f"+{i}" for i in range(1,7)], preds, linestyle='--', label='Forecast')
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("Not enough data to forecast. Add at least 2 months of expenses.")

    elif menu == "🌟 Goals":
        st.subheader("Set Financial Goals")
        with st.form("goal_form"):
            desc = st.text_input("Goal Description")
            target = st.number_input("Target Amount", step=100.0)
            current = st.number_input("Current Amount", step=100.0)
            deadline = st.date_input("Deadline")
            if st.form_submit_button("Add Goal"):
                add_goal(desc, target, current, str(deadline))
                st.success("Goal added.")
        df = pd.read_sql("SELECT * FROM Goals", sqlite3.connect(DB_PATH))
        if not df.empty:
            st.dataframe(df)

        # Debt Simulator Tab
    elif menu == "📆 Debt Simulator":
        st.subheader("📆 Debt Payoff Simulator")
    
        df_debts = get_debts()
    
        if df_debts.empty:
            st.warning("Please add debt entries first in the '💰 Manage Debts' section.")
        else:
            method = st.selectbox("Select payoff method", ["Snowball", "Avalanche"])
    
            st.info(f"Using {method} strategy to simulate debt payoff...")
    
            payoff_df = debt_payoff_simulator(df_debts, method.lower())
    
            if payoff_df.empty:
                st.error("Could not calculate payoff — ensure debts have balance, interest rate, and minimum payment.")
            else:
                st.dataframe(payoff_df)


    elif menu == "💰 Budgets":
        st.subheader("Monthly Budgets")
        with st.form("budget_form"):
            cat = st.selectbox("Category", get_categories("Expense"))
            month = st.text_input("Month (YYYY-MM)")
            amount = st.number_input("Budget Amount", step=100.0)
            if st.form_submit_button("Set Budget"):
                add_budget(cat, month, amount)
                st.success("Budget set.")
        df = pd.read_sql("SELECT * FROM Budgets", sqlite3.connect(DB_PATH))
        if not df.empty:
            st.dataframe(df)

    elif menu == "🌐 Manage Categories":
        st.subheader("Manage Categories")
        with st.form("cat_form"):
            name = st.text_input("Category Name")
            type_ = st.selectbox("Type", ["Income", "Expense", "Debt"])
            if st.form_submit_button("Add Category"):
                add_category(name, type_)
                st.success("Category added.")
        df = pd.read_sql("SELECT * FROM Categories", sqlite3.connect(DB_PATH))
        if not df.empty:
            st.dataframe(df)

if __name__ == '__main__':
    main()
