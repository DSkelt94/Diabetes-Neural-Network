# Week 6 - Diabetes Detection Neural Network
# Improt the libraries
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os


# Get the data
def get_data():
    # Get the path to the data
    # Get the current path
    path_current = os.getcwd()

    # Get the parent path
    path_parent = os.path.dirname(path_current)

    # Get the data path
    path_data = os.path.join(path_parent, "Data")

    # Join the path to the data
    data = pd.read_csv(os.path.join(path_data, "diabetes.csv"))

    # Get the data
    X = data.iloc[:, 2:13].values # Get all the rows and columns 2 to 12
    y = data.iloc[:, 13].values 


    # Replace the categorical data with numbers
    df_X = pd.DataFrame(X, columns= ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI'])
    # Male : 0, Female : 1
    df_X['Gender'].replace(['M', 'F'], [0, 1], inplace=True)
    # Normal : 0, Pre-Diabetic : 1, Diabetic : 2
    df_y = pd.DataFrame(y, columns=['CLASS'])
    df_y['CLASS'].replace(['N', 'P', 'Y'], [0, 1, 2], inplace=True)

    # Covert data to float
    # df_X = df_X.astype(float)

    # Show the first 5 data sets
    print(df_X.head(5))

    return df_X, df_y


# Split the data into training and testing
def split_data(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, layers, iterations, activation, solver):
    # Create the model
    model = MLPClassifier(hidden_layer_sizes=(layers,), max_iter=iterations, activation=activation, solver = solver, random_state=0)

    # Fit the model
    model.fit(X_train, y_train.values.ravel())

    # Print Results
    print("Test score accuracy = {}".format(accuracy_score(y_test, model.predict(X_test))))
    print("Training score accuracy = {}".format(accuracy_score(y_train, model.predict(X_train))))
    print("Confusion Matrix = \n {}".format(confusion_matrix(y_test, model.predict(X_test))))

    return accuracy_score(y_test, model.predict(X_test))

def main():
    # Get the data
    X, y = get_data()
    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']
    best = []
    best_model = [0, 0, '', '']

    # Show the data
    print(y)
    print("\n")

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("y_train = {}".format(len(y_train)))
    print("X_train = {}".format(len(X_train)))
    print("\n")

    # Create the model and iterate through the layers, activation, and solver
    for j in activations:
        for k in solvers:
            highest_accuracy = 0
            for i in range(1, 30):
                print("Layers = {}".format(i))
                print("Activation = {}".format(j))
                print("Solver = {}".format(k))
                accuracy = train_model(X_train, y_train, X_test, y_test, i, 2000, j, k)
                print("\n")

                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_layers = i
                    best_activation = j
                    best_solver = k
                if accuracy > best_model[0]:
                    best_model[0] = accuracy
                    best_model[1] = i
                    best_model[2] = j
                    best_model[3] = k

            best.append("Best Accuracy = {} \n Best Layer Amount = {}, \n Best Activation {}, \n Best Solver = {}".format(highest_accuracy, best_layers, best_activation, best_solver))

    print("Best Model:")
    print("Best Accuracy = {} \n Best Layer Amount = {}, \n Best Activation = {}, \n Best Solver = {}".format(best_model[0], best_model[1], best_model[2], best_model[3]))


if __name__ == "__main__":
    main()