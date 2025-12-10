# 💰 AI Finance Tracker

An intelligent personal finance tracking application powered by TensorFlow and Streamlit. Automatically categorizes your expenses using machine learning and provides beautiful visualizations of your spending habits.

## ✨ Features

- 🤖 **AI-Powered Categorization**: Automatically categorizes transactions using a trained TensorFlow Lite model
- 💵 **Income Detection**: Smart keyword-based detection for income transactions (salary, commission, etc.)
- 📊 **Interactive Dashboard**: Real-time overview of income, expenses, and net balance
- 📈 **Visual Analytics**: Beautiful charts and graphs powered by Plotly
- ✏️ **Easy Editing**: Inline data editor for managing transactions
- 🔍 **Advanced Filtering**: Filter by date range, category, and transaction type
- 💾 **SQLite Database**: Persistent local storage for all your financial data

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

The app will open automatically in your browser at `http://localhost:8501`

## 📖 Usage

### Adding Transactions
Simply enter transactions in natural language:
- `500rs coffee` → Expense
- `30000rs salary` → Income 💰
- `1200rs petrol` → Expense

The AI will automatically categorize your transaction!

### Categories
The app recognizes these categories:
- Education, Entertainment, Essentials
- Food (Friends), Food (Self), Food (Snacks)
- Gifts, Luxuries, Petrol, Rent
- Miscellaneous (Others), Miscellaneous (Self)

### Income Keywords
Transactions containing these keywords are marked as income:
`salary`, `commission`, `income`, `deposit`, `refund`, `bonus`, `freelance`, `consulting`, `dividend`, `interest`, `cashback`, `reimbursement`

## 🛠️ Training Your Own Model (Optional)

If you want to train the model with your own data:

1. Prepare your data in `expenses.csv` with columns: `name` and `category`
2. Run data preparation:
   ```bash
   python dataprep01.py
   ```
3. Train the model:
   ```bash
   python dataprep02.py
   ```

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application
├── database_setup.py       # Database initialization
├── db_service.py          # Database operations
├── run_app_sim.py         # AI prediction logic
├── dataprep01.py          # Data preprocessing
├── dataprep02.py          # Model training
├── income_config.py       # Income detection configuration
├── requirements.txt       # Python dependencies
├── processed_data/        # Trained model files
│   ├── expense_category_model.tflite
│   ├── tokenizer_config.json
│   └── category_map.json
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Note

This app stores financial data locally in an SQLite database. Your data never leaves your computer. The `.gitignore` file is configured to exclude your personal financial data from version control.
