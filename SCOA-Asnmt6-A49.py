# Import necessary libraries
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define fuzzy variables and membership functions
rooms = ctrl.Antecedent(np.arange(0, 10, 1), 'rooms')
size = ctrl.Antecedent(np.arange(0, 5000, 100), 'size')
age = ctrl.Antecedent(np.arange(0, 52, 1), 'age')
price = ctrl.Consequent(np.arange(0, 500001, 10000), 'price')

# Membership functions for rooms
rooms['few'] = fuzz.trimf(rooms.universe, [0, 0, 5])
rooms['average'] = fuzz.trimf(rooms.universe, [2, 5, 8])
rooms['many'] = fuzz.trimf(rooms.universe, [5, 10, 10])

# Membership functions for size
size['small'] = fuzz.trimf(size.universe, [0, 0, 2000])
size['medium'] = fuzz.trimf(size.universe, [1000, 2500, 4000])
size['large'] = fuzz.trimf(size.universe, [3000, 5000, 5000])

# Membership functions for age
age['new'] = fuzz.trimf(age.universe, [0, 0, 20])
age['moderate'] = fuzz.trimf(age.universe, [10, 25, 40])
age['old'] = fuzz.trimf(age.universe, [30, 52, 52])

# Membership functions for price
price['low'] = fuzz.trimf(price.universe, [0, 0, 200000])
price['medium'] = fuzz.trimf(price.universe, [100000, 300000, 500000])
price['high'] = fuzz.trimf(price.universe, [300000, 500000, 500000])

# Define fuzzy rules
rule1 = ctrl.Rule(rooms['few'] & size['small'] & age['new'], price['low'])
rule2 = ctrl.Rule(rooms['average'] & size['medium'] & age['moderate'], price['medium'])
rule3 = ctrl.Rule(rooms['many'] & size['large'] & age['old'], price['high'])
rule4 = ctrl.Rule(rooms['few'] & size['medium'] & age['old'], price['low'])
rule5 = ctrl.Rule(rooms['many'] & size['small'] & age['new'], price['medium'])
rule6 = ctrl.Rule(rooms['average'] & size['large'] & age['moderate'], price['high'])

# Create control system and simulation
fuzzy_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
fuzzy_control_system_simulation = ctrl.ControlSystemSimulation(fuzzy_control_system)

# Neural network model
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Regression output

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=16)

# Evaluate the model
y_pred_nn = model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f"Neural Network MSE: {mse_nn}")

# Example input values for fuzzy system
rooms_input = 6
size_input = 3000
age_input = 20

# Using fuzzy control system to get a price estimate
fuzzy_control_system_simulation.input['rooms'] = rooms_input
fuzzy_control_system_simulation.input['size'] = size_input
fuzzy_control_system_simulation.input['age'] = age_input
fuzzy_control_system_simulation.compute()
price_output_fuzzy = fuzzy_control_system_simulation.output['price']
print("Fuzzy Logic Estimated Price:", price_output_fuzzy)

# Prepare input for neural network
input_data_nn = scaler.transform([[rooms_input, size_input, age_input, 0, 0, 0, 0, 0]])  # Adjust to match feature dimensions
price_output_nn = model.predict(input_data_nn)
print("Neural Network Estimated Price:", price_output_nn[0][0])
