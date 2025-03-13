# ApnaBank - AI-Powered Banking Web App with OCR

## Overview
ApnaBank is a full-stack web application designed as a modern banking platform with AI-driven features for loan eligibility prediction, CIBIL score analysis, and financial data visualization.

## Features
- **Homepage:** Interactive landing page with smooth scrolling animations and banking service details.
- **User Authentication:** Secure login and registration system.
- **Dashboard:** Displays monthly transactions and financial data visualizations (pie charts, bar graphs, etc.).
- **Loan Checker:** Users can upload recent payslips, and the system extracts data using OCR and a trained model to determine loan eligibility and CIBIL score.
- **CIBIL Score Predictor:** Predicts the CIBIL score using a pre-trained ML model.
- **Expense Tracker:** Tracks personal expenses based on categories and transactions.
- **AI Integration:** Machine learning models to process financial data and determine loan eligibility.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Database:** SQLAlchemy
- **Machine Learning:** scikit-learn, pandas, numpy
- **OCR Processing:** OpenCV, Tesseract

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ApnaBank.git
   cd ApnaBank
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   streamlit run apnabank_app.py
   ```

## Dataset Information
ApnaBank uses two datasets:
1. **user_data.csv** - Contains user details like expense category, income, etc.
2. **transaction_data.csv** - Includes transaction history with transaction amount, date, and category.

## AI Model for CIBIL Score Prediction
- A machine learning model (`cibil_model.pkl`) is trained using the transaction dataset.
- If a user is from a **Tier-1 college (IIT, NIT)** or a **top ranker**, they can still qualify for loans even with a low CIBIL score.

## Project Structure
```
ApnaBank/
├── 
│     ├── apnabank_app.py
│     ├── apnabank.db
│     ├── cibil_model.py
│     ├── user_data.csv
│     ├── transaction_data.csv
│     └── venv/
│   
└── README.md
```

## Flowchart - Loan Eligibility Process
![Loan Flowchart](Screenshot_2025-03-13_221538.png)

## Contributing
Feel free to contribute by submitting pull requests or reporting issues.

## License
MIT License

