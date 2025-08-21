Supermart Grocery Dataset Analysis
📌 Project Overview

This project analyzes Supermart grocery sales data to gain insights into sales performance, trends, and patterns.
It uses Python, Pandas, Seaborn, Matplotlib, and Machine Learning techniques to clean, visualize, and model the data for better decision-making.

🧩 Features

Data Cleaning & Preprocessing

Handles missing values and removes duplicates.

Converts date columns to datetime format.

Encodes categorical variables for modeling.

Exploratory Data Analysis (EDA)

Sales distribution by category and region.

Monthly, yearly, and city-wise sales analysis.

Correlation heatmaps and trend visualizations.

Machine Learning

Trains a Linear Regression model to predict sales.

Evaluates model performance using MSE and R² score.

Visualizations

Sales by category, month, and year.

Top cities by sales.

Correlation heatmaps and actual vs predicted sales comparisons.

🗂️ Project Structure
📦 Supermart_Grocery_Analysis
 ┣ 📜 main.py                  # Main project script
 ┣ 📂 data
 ┃ ┗ supermart_grocery_dataset.csv   # Raw dataset
 ┣ 📜 README.md               # Project documentation
 ┗ 📜 requirements.txt        # Dependencies list

🛠️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/supermart-grocery-analysis.git
cd supermart-grocery-analysis

2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

📦 Dependencies

Include the following in requirements.txt:

pandas
numpy
matplotlib
seaborn
scikit-learn

🚀 Usage
Run the script
python main.py

Output

Displays sales trends, insights, and predictions.

Visualizations include bar charts, pie charts, and line charts for better understanding.

📊 Key Insights

Monthly and yearly sales trends identified.

Top-performing categories and cities recognized.

Linear Regression applied to predict future sales.

Correlation heatmaps reveal feature relationships.

🧠 Future Enhancements

Deploy an interactive dashboard using Streamlit or Plotly Dash.

Improve predictions using advanced machine learning models.

Add inventory forecasting and customer segmentation analysis.

👨‍💻 Author

Praveen Dane
B.Tech 4th Year | Aspiring Data Analyst & Python Developer