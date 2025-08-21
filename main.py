
# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 

#Load the dataset
 # Load the dataset
data = pd.read_csv('supermart_grocery_dataset.csv')
 # Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())
 # Drop any rows with missing values
data.dropna(inplace=True)
 # Check for duplicates
data.drop_duplicates(inplace=True)

# Convert 'Order Date' to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'])
 # Extract day, month, and year from 'Order Date'
data['Order Day'] = data['Order Date'].dt.day
data['Order Month'] = data['Order Date'].dt.month
data['Order Year'] = data['Order Date'].dt.year

# Initialize the label encoder
le = LabelEncoder()
 # Encode categorical variables
data['Category'] = le.fit_transform(data['Category'])
data['Sub Category'] = le.fit_transform(data['Sub Category'])
data['City'] = le.fit_transform(data['City'])
data['Region'] = le.fit_transform(data['Region'])
data['State'] = le.fit_transform(data['State'])
data['Order Month'] = le.fit_transform(data['Order Month'])
 # Display the first few rows after encoding
print(data.head())


#  Exploratory Data Analysis (EDA)
#  Distribution of Sales by Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Sales', data=data, palette='Set2')
plt.title('Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()


# Sales Trends Over Time
plt.figure(figsize=(12, 6))
data.groupby('Order Date')['Sales'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

#  Correlation Heatmap
plt.figure(figsize=(12, 8))

corr_matrix = data.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#  Select features and target variable
features = data.drop(columns=['OrderID', 'CustomerName','Order Date', 'Sales', 'Order Month'])
target = data['Sales']
 # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2, random_state=42)
 # Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression Model
 # Initialize the model
model = LinearRegression()
 # Train the model
model.fit(X_train, y_train)
 # Make predictions
y_pred = model.predict(X_test)

 # Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#  Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)], color='red')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

df = pd.read_csv('supermart_grocery_dataset.csv')

df.head()

df.info()

 #Let's change the datatype of Order Date from object to date
df ['Order Date'] = pd.to_datetime (df ['Order Date'],
errors='ignore')

df.info()

da=df.groupby("Category")
da.first()

 # firstly, we group by Category and get the total number of sales for each category
Sales_category=df.groupby("Category")["Sales"].sum()
 #we create a plot of sales by category
Sales_category.plot(kind='bar')
plt.title('Category by Sales', fontsize = 14)
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()

#Extract month from the order date
df['month_no'] = df['Order Date'].dt.month
df['Month'] = pd.to_datetime(df['Order Date']).dt.strftime('%B')
df['year'] = df['Order Date'].dt.year

df.head()

# Sum up sales by month
monthly_sales=df.groupby('Month')['Sales'].sum().reset_index()
 # Sort the data by month
monthly_sales_sorted=monthly_sales.sort_values(by='Month')
 # Create the line chart
plt.figure(figsize=(10,6))
plt.plot(monthly_sales_sorted['Month'],
monthly_sales_sorted['Sales'],marker='o')
plt.title('SalesbyMonth')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(monthly_sales_sorted['Month'],['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.grid(True)
plt.show()


Yearly_Sales=df.groupby("year")["Sales"].sum()
 # we create a pie chart with the sales by year
plt.pie(Yearly_Sales, labels=Yearly_Sales.index,
autopct='%1.1f%%')
plt.title('Sales by Year')
plt.show()
# Monthly_Sales.plot(kind='pie')
plt.title('Yearly Sales', fontsize = 14)
plt.show()


city_sales = df[['City', 'Sales']]
 # Step 2: Calculate total sales per city
total_sales = city_sales.groupby('City').sum()
 # Step 3: Sort the cities by sales
sorted_cities = total_sales.sort_values(by='Sales',
ascending=False)
 # Step 4: Select the top 5 cities
top_cities = sorted_cities.head(5)
 # Step 5: Plot the bar chart
plt.bar(top_cities.index, top_cities['Sales'])
plt.xlabel('City')
plt.ylabel('Sales')
plt.title('Top 5 Cities by Sales')
plt.xticks(rotation=45)
plt.show()
