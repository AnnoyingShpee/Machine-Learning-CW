# STUDENT_ID: 100276935
# Created on: 2022-11-14
# Last update: 2022-12-13
# Description:  Applies different parameters to the Decision Tree, Random Forest, and KNN Classifiers and calculates
#               the median accuracy of each model with different parameter combinations. The model from each classifier
#               type with the best accuracy is used to predict the test data. The median accuracy score of each model
#               and the best parameters for the best model are written in optimisation_evaluation.txt. The median
#               accuracy scores are also plotted for each classifier and saved as images named DT_Accuracy_score.png,
#               RF_Accuracy_Score.png, and KNN_Accuracy_Score.png


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

random_seed = 1  # Random Generator Seed
k_folds = 5  # number of folds
scoring_options = "accuracy"  # Type of scoring to get from classifiers

# load data, x is an array with all instances with attributes, and y is an array with the target attribute
x, y = load_breast_cancer(return_X_y=True)

# split data into a train and a test set
# test_size: 70% instances training, 30% instances testing
split_function = ShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)
instance_index_list = list(split_function.split(x, y))
train_index_set = instance_index_list[0][0]
test_index_set = instance_index_list[0][1]

# get training/testing instances from the source dataset
x_train = x[train_index_set, :]
y_train = y[train_index_set]
x_test = x[test_index_set, :]
y_test = y[test_index_set]

# Array for DT classifier parameters
dt_classifier_params = [2, 3, 5, 7, 10]
# Array for RF classifier parameters
rf_classifier_params = [100, 200, 500]
# Array for KNN classifier parameters
knn_classifier_params = [[1, 11, 21, 31, 51], ["euclidean", "manhattan"]]

# knn_classifier_params = {
#     "k": [1, 11, 21, 31, 51],
#     "metric": ["euclidean", "manhattan"]
# }
# Get all possible combinations of KNN parameters of dictionary
# knn_params = hf.get_combinations(knn_classifier_params)


# k-fold validator object
k_fold = KFold(n_splits=k_folds, shuffle=False)
# Results of the script will be printed and written to optimisation_evaluation.txt
x_plots = []
y_plots = []
best_params = {}
best_score = 0
best_classifier = None

try:
    # Open file to write
    f = open("../output/optimisation_evaluation.txt", "w")

    print("==========================================================")
    f.write("==========================================================\n")
    print("5-fold validation on Decision Tree Classifiers")
    f.write("5-fold validation on Decision Tree Classifiers \n")
    print("==========================================================")
    f.write("==========================================================\n")
    # Loop through each parameter
    for max_depth in dt_classifier_params:
        print(f"max_depth = {max_depth}")
        f.write(f"max_depth = {max_depth}\n")
        # Create Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=random_seed)
        # Cross validate classifier to get array of accuracy values (5 folds = 5 elements)
        dt_cross_val = cross_val_score(dt_classifier, x_train, y_train, cv=k_fold, scoring=scoring_options)
        # Get median value of accuracy list
        median_value = np.median(dt_cross_val)
        # Add classifier parameters to x-axis plot array
        x_plots.append(max_depth)
        # Add median accuracy value to y-axit plot array
        y_plots.append(median_value)
        print(f"Median accuracy score= {median_value}")
        f.write(f"Median accuracy score= {median_value} \n")
        # Find the best accuracy score among all the different-parameter classifiers and save the parameter and
        # classifier
        if median_value > best_score:
            best_score = median_value
            best_params["max_depth"] = max_depth
            best_classifier = dt_classifier
        print("-----------------------------------------------------------")
        f.write("-----------------------------------------------------------\n")
    # Plot a graph of median accuracy values of all Decision Tree classifiers with different max_depth
    plt.figure(1)
    plt.plot(x_plots, y_plots, marker="o")
    plt.title("DT Median Accuracy Score with Criterion = Entropy")
    plt.xlabel("max_depth")
    plt.ylabel("Median Accuracy Score")
    # Save the plot as a PNG image
    plt.savefig("../output/DT_Accuracy_Score.png")
    # plt.show()
    print(f"Best DT accuracy score: {best_score} ; Best DT params: {best_params} ")
    f.write(f"Best DT accuracy score: {best_score} ; Best DT params: {best_params} \n")
    # Train the best-parameter classifier and predict the test data
    best_classifier.fit(x_train, y_train)
    dt_predictions = best_classifier.predict(x_test)
    # Get the accuracy score from test label and predictions
    dt_acc_score = accuracy_score(y_test, dt_predictions)
    print(f"Best DT Params Test Data Accuracy Score = {dt_acc_score}")
    f.write(f"Best DT Params Test Data Accuracy Score = {dt_acc_score} \n")
    print("==========================================================\n")
    f.write("==========================================================\n\n")

    print("==========================================================")
    f.write("==========================================================\n")
    print("5-fold validation on Random Forest Classifiers")
    f.write("5-fold validation on Random Forest Classifiers \n")
    print("==========================================================")
    f.write("==========================================================\n")
    # Clear the variables
    x_plots = []
    y_plots = []
    best_params = {}
    best_score = 0
    best_classifier = None
    # Loop through each parameter
    for n_estimators in rf_classifier_params:
        print(f"n_estimators = {n_estimators}")
        f.write(f"n_estimators = {n_estimators}\n")
        # Create Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed)
        # Cross validate classifier to get array of accuracy values (5 folds = 5 elements)
        rf_cross_val = cross_val_score(rf_classifier, x_train, y_train, cv=k_fold, scoring=scoring_options)
        # Get median value of accuracy list
        median_value = np.median(rf_cross_val)
        # Add classifier parameters to x-axis array
        x_plots.append(n_estimators)
        # Add median accuracy value to y-axis array
        y_plots.append(median_value)
        print(f"Median accuracy score= {median_value}")
        f.write(f"Median accuracy score= {median_value} \n")
        # Find the best accuracy score among all the different-parameter classifiers and save the parameter and
        # classifier
        if median_value > best_score:
            best_score = median_value
            best_params["n_estimators"] = n_estimators
            best_classifier = rf_classifier
        print("-----------------------------------------------------------")
        f.write("-----------------------------------------------------------\n")
    # Plot a graph of median accuracy values of all Random Forest classifiers with different n_estimators
    plt.figure(2)
    plt.plot(x_plots, y_plots, marker="o")
    plt.title("RF Median Accuracy Score with Random State = 1")
    plt.xlabel("n_estimators")
    plt.ylabel("Median Accuracy Score")
    # Save the plot as a PNG image
    plt.savefig("../output/RF_Accuracy_Score.png")
    # plt.show()
    print(f"Best RF accuracy score: {best_score} ; Best RF params: {best_params}")
    f.write(f"Best RF accuracy score: {best_score} ; Best RF params: {best_params} \n")
    # Train the best-parameter classifier and predict the test data
    best_classifier.fit(x_train, y_train)
    rf_predictions = best_classifier.predict(x_test)
    # Get the accuracy score from test label and predictions
    rf_acc_score = accuracy_score(y_test, rf_predictions)
    print(f"Best RF Params Test Data Accuracy Score = {rf_acc_score}")
    f.write(f"Best RF Params Test Data Accuracy Score = {rf_acc_score} \n")
    print("==========================================================\n")
    f.write("==========================================================\n\n")

    print("==========================================================")
    f.write("==========================================================\n")
    print("5-fold validation on KNN Classifiers")
    f.write("5-fold validation on KNN Classifiers \n")
    print("==========================================================")
    f.write("==========================================================\n")
    # Clear variables
    x_plots = []
    y_plots = []
    best_params = {}
    best_score = 0
    best_classifier = None
    # Loop through each parameter combinations
    for i in range(len(knn_classifier_params[0])):
        for j in range(len(knn_classifier_params[1])):
            k = knn_classifier_params[0][i]
            metric = knn_classifier_params[1][j]
            print(f"n_estimators = {k} : metric = {metric}")
            f.write(f"n_estimators = {k} : metric = {metric} \n")
            # Create KNN Classifier from accessed parameters
            knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
            # Cross validate classifier
            knn_cross_val = cross_val_score(knn_classifier, x_train, y_train, cv=k_fold, scoring=scoring_options)
            # Get median value of accuracy list
            median_value = np.median(knn_cross_val)
            # Checks the type of metric used for KNN Classifier's metric parameter and saves it as n_neighbours(metric)
            # in x-axis array.
            # euclidean = E , manhattan = M
            if metric == "euclidean":
                x_plots.append(f"{k}(E)")
            elif metric == "manhattan":
                x_plots.append(f"{k}(M)")
            # Add median accuracy value to y-axis array
            y_plots.append(median_value)
            print(f"Median accuracy score= {median_value}")
            f.write(f"Median accuracy score= {median_value} \n")
            # Find the best accuracy score among all the different-parameter classifiers and save the parameter and
            # classifier
            if median_value > best_score:
                best_score = median_value
                best_params["k"] = k
                best_params["metric"] = metric
                best_classifier = knn_classifier
            print("-----------------------------------------------------------")
            f.write("-----------------------------------------------------------\n")
    # Plot a graph of median accuracy values of all KNN classifiers with different n_neighbours and metrics
    plt.figure(3)
    plt.plot(x_plots, y_plots, marker="o")
    plt.title("KNN Median Accuracy Score")
    plt.xlabel("Parameters [k + (E)uclidean/(M)anhattan]")
    plt.ylabel("Median Accuracy Score")
    # Save plot as a PNG image
    plt.savefig("../output/KNN_Accuracy_Score.png")
    # plt.show()
    print(f"Best KNN accuracy score: {best_score} ; Best KNN params: {best_params}")
    f.write(f"Best KNN accuracy score: {best_score} ; Best KNN params: {best_params} \n")
    # Train the best-parameter classifier and predict the test data
    best_classifier.fit(x_train, y_train)
    knn_predictions = best_classifier.predict(x_test)
    # Get the accuracy score from test label and predictions
    knn_acc_score = accuracy_score(y_test, knn_predictions)
    print(f"Best KNN Params Test Data Accuracy Score = {knn_acc_score}")
    f.write(f"Best KNN Params Test Data Accuracy Score = {knn_acc_score} \n")
    print("==========================================================\n")
    f.write("==========================================================\n\n")

    f.close()
except Exception as e:
    print(e)

