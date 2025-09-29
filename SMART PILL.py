streamlit run your_app.py
import streamlit as st

# ===================================
# Simulate the Dataset
# ===================================
np.random.seed(42)
insurance = pd.DataFrame({
    'hour': np.random.randint(6, 22, 1000),
    'day_of_week': np.random.randint(0, 7, 1000),
    'missed_last_dose': np.random.randint(0, 2, 1000),
    'cognitive_impairment_level': np.random.choice(['low', 'medium', 'high'], 1000),
    'age_group': np.random.choice(['young', 'middle-aged', 'elderly'], 1000),
    'dose_taken': np.random.randint(0, 2, 1000)
})

# ===================================
# Feature Engineering
# ===================================
insurance_encoded = pd.get_dummies(insurance, columns=['cognitive_impairment_level', 'age_group'], drop_first=True)

X = insurance_encoded.drop('dose_taken', axis=1)
y = insurance_encoded['dose_taken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
