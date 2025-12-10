# Income Detection Configuration

# KEYWORD-BASED INCOME DETECTION (Currently Active)
# The app now detects income by looking for these keywords in your transaction text
# You can customize this list to match how you describe income transactions

INCOME_KEYWORDS = [
    'salary',
    'commission',
    'income',
    'deposit',
    'refund',
    'bonus',
    'payment received',
    'freelance',
    'consulting',
    'dividend',
    'interest',
    'cashback',
    'reimbursement'
]

# Examples of income transactions that will be detected:
# - "30000rs salary" ✓
# - "5000rs commission" ✓
# - "2000rs freelance payment" ✓
# - "500rs cashback" ✓

# CATEGORY-BASED INCOME DETECTION (For Future Use)
# When you retrain your ML model with income categories, add them here:
INCOME_CATEGORIES = []

# Note: Currently, your ML model only has expense categories:
# Education, Entertainment, Essentials, Food (Friends), Food (Self), 
# Food (Snacks), Gifts, Luxuries, Miscellaneous (Others), 
# Miscellaneous (Self), Petrol, Rent

# To add category-based income detection:
# 1. Add income transactions to your expenses.csv with appropriate categories
# 2. Re-run dataprep01.py and dataprep02.py to retrain the model
# 3. Update INCOME_CATEGORIES list above with the new income category names
