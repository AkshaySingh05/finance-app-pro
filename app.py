# finance_app_pro/app.py

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DB_PATH = "personal_finance.db"

import sqlite3

conn = sqlite3.connect("personal_finance.db")
cursor = conn.cursor()

# Convert all category types to lowercase
cursor.execute("UPDATE Categories SET type = LOWER(TRIM(type))")

conn.commit()
conn.close()

print("Category types normalized to lowercase.")

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
    params = ()
    if type_filter:
        query += " WHERE LOWER(TRIM(type)) = ?"
        params = (type_filter.lower().strip(),)
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    # Debug output
    if df.empty:
        print(f"[DEBUG] No categories found for type: '{type_filter}'")
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
    
    # Update debt balance if this is a Debt
    if type_ == "Debt":
        cursor = conn.cursor()
        cursor.execute("SELECT id, balance FROM Debts WHERE creditor = ?", (category,))
        result = cursor.fetchone()
        if result:
            debt_id, current_balance = result
            new_balance = max(current_balance - amount, 0)
            cursor.execute("UPDATE Debts SET balance = ? WHERE id = ?", (new_balance, debt_id))
    
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

def delete_transaction(txn_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM Transactions WHERE id = ?", (txn_id,))
    conn.commit()
    conn.close()

def update_transaction(txn_id, date, type_, category, amount, note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        UPDATE Transactions
        SET date = ?, type = ?, category = ?, amount = ?, note = ?
        WHERE id = ?
    ''', (str(date), type_, category, amount, note, txn_id))
    conn.commit()
    conn.close()


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
        st.subheader("➕ Add New Transaction")
        with st.form("txn_form"):
            date = st.date_input("Date")
            type_ = st.selectbox("Type", ["Income", "Expense", "Debt"])
            
            cat_type = type_.strip().lower()
            filtered_categories = get_categories(type_filter=cat_type)

            if not filtered_categories:
                st.warning("⚠ No categories found for this type. Please add them under 'Manage Categories'.")
                category = ""
            else:
                category = st.selectbox("Category", filtered_categories)
            
            amount = st.number_input("Amount", step=0.01)
            note = st.text_input("Note")
            
            if st.form_submit_button("Submit"):
                if category:
                    add_transaction(str(date), type_, category, amount, note)
                    st.success("Transaction added successfully!")
                else:
                    st.error("Please select a valid category.")

            st.write("DEBUG - Category Types in DB:")
            st.dataframe(pd.read_sql("SELECT DISTINCT type FROM Categories", sqlite3.connect(DB_PATH)))


        # 🔍 View and Manage Transactions
        st.markdown("---")
        st.subheader("📝 Edit or Delete Existing Transactions")
    
        df_transactions = get_transactions()
        if df_transactions.empty:
            st.info("No transactions found.")
        else:
            df_transactions['label'] = df_transactions.apply(
                lambda row: f"{row['id']} | {row['date']} | {row['type']} | R{row['amount']:.2f} | {row['note'] or row['category']}",
                axis=1
            )
            
            selected_txn = st.selectbox(
                "Select a transaction to edit or delete",
                df_transactions['label'].tolist()
            )
            txn_id = int(selected_txn.split(" | ")[0])
    
            txn_id = int(selected_txn.split(" | ")[0])
            txn_row = df_transactions[df_transactions['id'] == txn_id].iloc[0]
    
            with st.form("edit_transaction_form"):
                new_date = st.date_input("Edit Date", value=pd.to_datetime(txn_row['date']))
                new_type = st.selectbox("Edit Type", ["Income", "Expense", "Debt"], index=["Income", "Expense", "Debt"].index(txn_row['type']))
                new_category = st.selectbox("Edit Category", get_categories(), index=get_categories().index(txn_row['category']))
                new_amount = st.number_input("Edit Amount", value=txn_row['amount'], step=0.01)
                new_note = st.text_input("Edit Note", value=txn_row['note'])
    
                col1, col2 = st.columns(2)
                if col1.form_submit_button("Update"):
                    update_transaction(txn_id, new_date, new_type, new_category, new_amount, new_note)
                    st.success("✅ Transaction updated.")
                    st.stop()
    
                if col2.form_submit_button("Delete"):
                    delete_transaction(txn_id)
                    st.warning("⚠️ Transaction deleted.")
                    st.stop()  # Stop execution safely after deleting

    elif menu == "🏋️ KPIs & Dashboard":
        df = get_transactions()
        df['date'] = pd.to_datetime(df['date'])
        debts = get_debts()
        income = df[df['type'] == 'Income']['amount'].sum()
        expenses = df[df['type'] == 'Expense']['amount'].sum()
        debt_payments = df[df['type'] == 'Debt']['amount'].sum()
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

    elif menu == "📆 Debt Simulator":
                st.subheader("📆 Debt Payoff Simulator")
            
                # --- 1. Debt Entry Form ---
                st.markdown("### ➕ Add a New Debt")
                with st.form("add_debt_form"):
                    credit_limit = st.number_input("Credit Limit", min_value=0.0, step=0.01)
                    available_credit = st.number_input("Available Credit", min_value=0.0, step=0.01)
                    balance = credit_limit - available_credit
                
                    creditor = st.text_input("Creditor Name")
                    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.01)
                    min_payment = st.number_input("Minimum Monthly Payment", min_value=0.0, step=0.01)
                    due_date = st.date_input("Due Date (optional)")
                
                    submitted = st.form_submit_button("Save Debt")
                
                    if submitted:
                        add_debt(creditor, balance, interest_rate, str(due_date), min_payment, credit_limit)
                        st.success(f"✅ Debt for **{creditor}** saved successfully.")

            
                # --- 2. View Existing Debts ---
                df_debts = get_debts()
                if df_debts.empty:
                    st.warning("⚠ No debts found. Please add at least one entry to simulate payoff.")
                else:
                    st.markdown("### 🧾 Manage Debts")
                    
                    # Show editable table using st.expander
                    with st.expander("📝 Edit or Delete Existing Debts"):
                        debt_ids = df_debts['id'].tolist()
                        selected_debt = st.selectbox("Select a Debt to Edit/Delete", options=debt_ids, format_func=lambda x: df_debts[df_debts['id'] == x]['creditor'].values[0])
                    
                        if selected_debt:
                            debt_row = df_debts[df_debts['id'] == selected_debt].iloc[0]
                    
                            # Editable fields
                            creditor = st.text_input("Creditor Name", debt_row['creditor'])
                            balance = st.number_input("Outstanding Balance", value=float(debt_row['balance']), step=0.01)
                            interest_rate = st.number_input("Interest Rate (%)", value=float(debt_row['interest_rate']), step=0.01)
                            min_payment = st.number_input("Minimum Monthly Payment", value=float(debt_row['min_payment']), step=0.01)
                            credit_limit = st.number_input("Credit Limit", value=float(debt_row['credit_limit']), step=0.01)
                            due_date = st.date_input("Due Date", value=datetime.strptime(debt_row['due_date'], "%Y-%m-%d"))
                    
                            col1, col2 = st.columns(2)
                            with col1:
                                # New field to enter available balance and auto-calculate actual balance
                                available_balance = st.number_input("Available Balance", value=float(debt_row['credit_limit'] - debt_row['balance']), step=0.01)
                                computed_balance = credit_limit - available_balance
                                
                                if col1.button("💾 Update Debt"):
                                    try:
                                        conn = sqlite3.connect(DB_PATH)
                                        conn.execute("""
                                            UPDATE Debts SET creditor=?, balance=?, interest_rate=?, due_date=?, 
                                            min_payment=?, credit_limit=? WHERE id=?
                                        """, (creditor, computed_balance, interest_rate, str(due_date), min_payment, credit_limit, selected_debt))
                                        conn.commit()
                                        conn.close()
                                        st.success("✅ Debt updated successfully.")
                                        st.stop()
                                    except Exception as e:
                                        st.error(f"❌ Failed to update debt: {e}")

                    
                            with col2:
                                if st.button("❌ Delete Debt"):
                                    conn = sqlite3.connect(DB_PATH)
                                    conn.execute("DELETE FROM Debts WHERE id=?", (selected_debt,))
                                    conn.commit()
                                    conn.close()
                                    st.warning("🗑️ Debt deleted.")
                                    st.stop()

            
                    # --- 3. Select Payoff Method ---
                    st.markdown("### 🧠 Choose Payoff Method")
                    method = st.selectbox("Select method:", ["Snowball", "Avalanche"])
                    if st.button("Run Simulation"):
                        payoff_df = debt_payoff_simulator(df_debts.copy(), method.lower())
                        if not payoff_df.empty:
                            st.success("✅ Simulation complete. Here's your estimated payoff timeline:")
                            st.dataframe(payoff_df)
                        else:
                            st.warning("⚠ Simulation did not return any results.")


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
