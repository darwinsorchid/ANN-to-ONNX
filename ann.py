'''
@File_name: 'ann.py'
@Author: Alexandra Bekou
@Date created: 23-12-24 11:43am
@Description: Build and train an ANN model that predicts the possibility of a bank client leaving the bank depending on specific client's features.
'''
# Import libraries
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Read data
dataset = pd.read_csv('Churn_Modelling.csv')
# Features matrix
x = dataset.iloc[:, 3:-1]
# Dependent variable vector
y = dataset.iloc[:, -1]

print(x)
print(y)

# Encoding categorical data
# Gender column - binary
le = LabelEncoder()
x.iloc[:, 2] = le.fit_transform(x.iloc[:, 2])

print(x)

# Geography column - one-hot encoding 
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

print(x)

# Split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42)

# Apply feature scaling to all features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize ANN - basic fully connected neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# Compile model 
model.compile(optimizer = 'adam', # Stochastic Gradient Descent 
              loss = 'binary_crossentropy', # Binary classification / For non-binary classification: 'categorical_crossentropy' + softmax activation function
              metrics = ['accuracy'])

# Train model and specify hyperparameters
model.fit(x_train, y_train, batch_size = 32, epochs = 100)

# Use model for prediction - always 2D array expected by predict method
# False: client will most probably not leave 
# True: client will most probably leave
print(model.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Use model to make predictions on the test set
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5) # will be equal to 0 if under 0.5 else 1

# Vertically concatenate predicted vs real value vectors
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), (y_test.reshape(len(y_test), 1)), 1)))

# Confusion Matrix - visualize model accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Save model
model_path = 'saved_model'
tf.saved_model.save(model, model_path)
print(f"Model saved to {model_path}")

