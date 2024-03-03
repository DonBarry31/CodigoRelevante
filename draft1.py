import piplite
await piplite.install('skillsnetwork')
await piplite.install('seaborn')
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
warnings.filterwarnings("ignore", category=UserWarning) 
%matplotlib inline
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
await skillsnetwork.download(filepath,'./laptops.csv')
path = './laptops.csv'
df = pd.read_csv(path, header=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y= df[['Price']]
x= df[['CPU_frequency']]
lr.fit(x,y)
Yhat=lr.predict(x)
import seaborn as sns 
ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
mse = mean_squared_error(y, Yhat)
print(mse)
r2 = r2_score(y, Yhat)
print(r2)
r2_score_slr = lr.score(x, y)
print(r2_score_slr)
z = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU','Category']]
y = df[['Price']]
lr.fit(z,y)
lr.intercept_
Y_hat = lr.predict(z)
print(Y_hat)
ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
mse = mean_squared_error(y, Y_hat)
print(mse)
r2 = r2_score(y, Y_hat)
print(r2)
x = x.to_numpy().flatten()
f1 = np.polyfit(x, y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(x, y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(x, y, 5)
p5 = np.poly1d(f5)
X = df[['CPU_frequency']]
y = df['Price']

# Assuming degrees is a list containing the polynomial degrees
degrees = [1, 3, 5]

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model.predict(poly.fit_transform(x_new.reshape(-1, 1)))

    plt.plot(independent_variable, dependent_variable, '.', label='Actual data')
    plt.plot(x_new, y_new, '-', label=f'Degree {degree} Polynomial Fit', color='red')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')
    plt.legend()
    plt.show()

for degree in degrees:
    # Create PolynomialFeatures
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    # Plot the polynomial regression line on the test data
    PlotPolly(model, X_test, y_test, 'CPU_frequency')
X = df[['CPU_frequency']]
y = df['Price']

# Assuming degrees is a list containing the polynomial degrees
degrees = [1, 3, 5]

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model.predict(poly.fit_transform(x_new.reshape(-1, 1)))

    plt.plot(independent_variable, dependent_variable, '.', label='Actual data')
    plt.plot(x_new, y_new, '-', label=f'Degree {degree} Polynomial Fit', color='red')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')
    plt.legend()
    plt.show()

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
    PlotPolly(model, X_test, y_test, 'CPU_frequency')
    from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming df is your dataset with the selected features and the target variable 'Price'
X = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Step 1: StandardScaler for parameter scaling
    ('poly_features', PolynomialFeatures()),  # Step 2: PolynomialFeatures for generating polynomial features
    ('linear_reg', LinearRegression())     # Step 3: LinearRegression
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
