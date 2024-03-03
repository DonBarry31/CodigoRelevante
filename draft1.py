import skillsnetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning) 
import matplotlib.pyplot as plt 
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
#import asyncio
#import skillsnetwork

#async def download_file():
#    filepath = './laptops.csv'
#    await skillsnetwork.download(filepath, './laptops.csv')
#path = './laptops.csv'
df = pd.read_csv(filepath, header=0)

# Simple Linear Regression
lr = LinearRegression()
y_simple = df[['Price']]
x_simple = df[['CPU_frequency']]
lr.fit(x_simple, y_simple)
y_hat_simple = lr.predict(x_simple)

# Plot Simple Linear Regression
ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_simple, hist=False, color="b", label="Fitted Values", ax=ax1)
mse_simple = mean_squared_error(y_simple, y_hat_simple)
r2_simple = r2_score(y_simple, y_hat_simple)
r2_score_slr = lr.score(x_simple, y_simple)

# Multiple Linear Regression
features = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
x_multiple = df[features]
y_multiple = df[['Price']]
lr.fit(x_multiple, y_multiple)
y_hat_multiple = lr.predict(x_multiple)

# Plot Multiple Linear Regression
ax2 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_multiple, hist=False, color="b", label="Fitted Values", ax=ax2)
mse_multiple = mean_squared_error(y_multiple, y_hat_multiple)
r2_multiple = r2_score(y_multiple, y_hat_multiple)

# Polynomial Regression
X_poly = df[['CPU_frequency']]
degrees = [1, 3, 5]

def plot_polly(model, independent_variable, dependent_variable, name, degree):
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model.predict(poly.fit_transform(x_new.reshape(-1, 1)))

    plt.plot(independent_variable, dependent_variable, '.', label='Actual data')
    plt.plot(x_new, y_new, '-', label=f'Degree {degree} Polynomial Fit', color='red')
    plt.title(f'Polynomial Fit for Price ~ {name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(name)
    plt.ylabel('Price of laptops')
    plt.legend()
    plt.show()

for degree in degrees:
    # Create PolynomialFeatures
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(x_multiple)
    X_poly_test = poly.transform(x_multiple)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_train, y_multiple)

    # Plot the polynomial regression line on the test data
    plot_polly(model, X_poly, y_multiple, 'CPU_frequency', degree)

# Print results
print("\nSimple Linear Regression:")
print(f"MSE: {mse_simple}")
print(f"R-squared: {r2_simple}")
print(f"R-squared Score: {r2_score_slr}")

print("\nMultiple Linear Regression:")
print(f"MSE: {mse_multiple}")
print(f"R-squared: {r2_multiple}")

def plot_polly(model, independent_variable, dependent_variable, name, degree):
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model.predict(poly.fit_transform(x_new.reshape(-1, 1)))

    plt.plot(independent_variable, dependent_variable, '.', label='Actual data')
    plt.plot(x_new, y_new, '-', label=f'Degree {degree} Polynomial Fit', color='red')
    plt.title(f'Polynomial Fit for Price ~ {name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(name)
    plt.ylabel('Price of laptops')
    plt.legend()
    plt.show()

# Assuming df is your dataset with the selected features and the target variable 'Price'
X = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: StandardScaler for parameter scaling
    ('poly_features', PolynomialFeatures()),  # Step 2: PolynomialFeatures for generating polynomial features
    ('linear_reg', LinearRegression())  # Step 3: LinearRegression
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Print R² and MSE values
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")

# Plot polynomial regression lines on the test data for different degrees
degrees = [1, 3, 5]
for degree in degrees:
    # Create PolynomialFeatures
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_poly_test)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Print R² and MSE values
    print(f"\nDegree {degree} Polynomial Fit:")
    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")

    # Plot the polynomial regression line on the test data
    plot_polly(model, X_test, y_test, 'CPU_frequency', degree)