# Placeholder for final app
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
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        type TEXT,
        category TEXT,
        amount REAL,
        note TEXT
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        month TEXT,
        budget_amount REAL
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Debts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        creditor TEXT,
        balance REAL,
        interest_rate REAL,
        due_date TEXT,
        min_payment REAL,
        credit_limit REAL,
        initial_balance REAL
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        target_amount REAL,
        current_amount REAL,
        deadline TEXT
    )''')
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df_tx = pd.read_sql("SELECT * FROM Transactions", conn)
    df_tx["date"] = pd.to_datetime(df_tx["date"])
    df_debts = pd.read_sql("SELECT * FROM Debts", conn)
    df_budgets = pd.read_sql("SELECT * FROM Budgets", conn)
    df_goals = pd.read_sql("SELECT * FROM Goals", conn)
    conn.close()
    return df_tx, df_debts, df_budgets, df_goals

def add_transaction(date, type_, category, amount, note):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Transactions (date, type, category, amount, note) VALUES (?, ?, ?, ?, ?)",
                   (date, type_, category, amount, note))
    conn.commit()
    conn.close()

def kpi_metrics(df_tx, df_debts):
    income = df_tx[df_tx["type"] == "Income"]["amount"].sum()
    expenses = df_tx[df_tx["type"] == "Expense"]["amount"].sum()
    savings = income - expenses
    debt_payments = df_tx[df_tx["type"] == "Debt Payment"]["amount"].sum()

    savings_rate = (savings / income) * 100 if income > 0 else 0
    expense_ratio = (expenses / income) * 100 if income > 0 else 0
    net_cash_flow = savings - debt_payments

    # Credit Utilization & Debt Reduction
    if not df_debts.empty:
        df_debts["credit_util"] = df_debts.apply(
            lambda x: (x["balance"] / x["credit_limit"]) * 100 if x["credit_limit"] > 0 else 0, axis=1
        )
        df_debts["debt_reduction"] = df_debts.apply(
            lambda x: ((x["initial_balance"] - x["balance"]) / x["initial_balance"]) * 100 if x["initial_balance"] > 0 else 0, axis=1
        )
        avg_util = df_debts["credit_util"].mean()
        avg_reduction = df_debts["debt_reduction"].mean()
    else:
        avg_util = 0
        avg_reduction = 0

    # Emergency Fund (Monthly Expenses)
    emg = df_tx[(df_tx["type"] == "Income") & (df_tx["category"] == "Emergency Fund")]["amount"].sum()
    monthly_exp = df_tx[df_tx["type"] == "Expense"].groupby(df_tx["date"].dt.to_period("M"))["amount"].sum().mean()
    emergency_months = emg / monthly_exp if monthly_exp > 0 else 0

    return {
        "Savings Rate (%)": savings_rate,
        "Expense Ratio (%)": expense_ratio,
        "Net Cash Flow": net_cash_flow,
        "Avg Credit Utilization (%)": avg_util,
        "Avg Debt Reduction (%)": avg_reduction,
        "Emergency Fund Coverage (months)": emergency_months
    }

def main():
    st.set_page_config("Finance App Pro", layout="wide")
    init_db()
    df_tx, df_debts, df_budgets, df_goals = load_data()

    st.title("💼 Finance App – Pro Edition")

    tabs = st.tabs(["➕ Add Transaction", "📊 KPIs & Dashboard", "📈 Forecasting", "🎯 Goals", "💳 Debt Simulator", "💰 Budgets"])

    with tabs[0]:
        st.header("➕ Add New Transaction")
        with st.form("tx_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            date = col1.date_input("Date", datetime.today())
            type_ = col2.selectbox("Type", ["Income", "Expense", "Debt Payment"])
            category = col3.text_input("Category")
            amount = st.number_input("Amount", min_value=0.0)
            note = st.text_input("Note (optional)")
            submitted = st.form_submit_button("Add Transaction")
            if submitted:
                add_transaction(date.strftime("%Y-%m-%d"), type_, category, amount, note)
                st.success("Transaction added.")

    with tabs[1]:
        st.header("📊 Key Financial Indicators")
        kpis = kpi_metrics(df_tx, df_debts)
        for i, (k, v) in enumerate(kpis.items()):
            st.metric(label=k, value=f"{v:,.2f}" if "Cash" in k else f"{v:.1f}%")

    with tabs[2]:
        st.header("📈 Forecasting Trends")
        df = df_tx[df_tx["type"] == "Expense"].copy()
        if not df.empty:
            df = df.groupby(df["date"].dt.to_period("M")).sum().reset_index()
            df["month_index"] = range(len(df))
            model = LinearRegression()
            model.fit(df[["month_index"]], df["amount"])
            future = pd.DataFrame({"month_index": range(len(df), len(df)+6)})
            forecast = model.predict(future)

            plt.figure(figsize=(8, 3))
            plt.plot(df["month_index"], df["amount"], label="Actual")
            plt.plot(future["month_index"], forecast, linestyle="--", label="Forecast")
            plt.title("6-Month Expense Forecast")
            plt.legend()
            st.pyplot(plt)
        else:
            st.info("No data available for forecasting.")

if __name__ == "__main__":
    main()
.py
