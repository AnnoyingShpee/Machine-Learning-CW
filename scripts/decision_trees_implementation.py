# STUDENT_ID: 100276935
# Created on: 2022-11-04
# Last update: 2022-12-13, [last modification to your code]
# Description:  Uses the DecisionTreeClassifier to predict data and calculates the accuracy of the classifier
#               when the max_depth parameter is changed. Results are written to file dt_balanced_acc.txt and
#               dt_balanced_acc_scores.txt. Balanced Accuracy Scores are plotted out and saved in
#               dt_balanced_acc_scores.png and the Decision Tree with the best score gets plotted and saved in
#               dt_entropy_max_depth_{value}.png.


import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

import helper_functions

random_seed = 1

file_path = '../data/breast-cancer.data'
# Check data and remove rows containing ? or missing data
helper_functions.check_data(file_path)

# Loads data from csv into variable
csv_data = pandas.read_csv('../data/breast-cancer.data', header=None)

# Convert categorical (string) attribute values to numerical (integer)
data, unique_attr = helper_functions.category_to_numerical(csv_data)
# Split the data into training and test sets
x_train, y_train, x_test, y_test = helper_functions.split_data(data, 0.3, random_seed)

# Build Decision Tree Classifier with max depth of 3 using entropy, random state set constant
decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=random_seed)
# Train the classifier with training data set
decision_tree_classifier.fit(x_train, y_train)
# Predict the class of the training set
train_predictions = decision_tree_classifier.predict(x_train)
# Get the Balanced Accuracy Score of the training set predictions and print
train_acc = balanced_accuracy_score(y_train, train_predictions)
print(f"dt_balanced_acc_train = {train_acc} \n")
# Get the Balanced Accuracy Score of the test set predictions and print
test_predictions = decision_tree_classifier.predict(x_test)
test_acc = balanced_accuracy_score(y_test, test_predictions)
print(f"dt_balanced_acc_test = {test_acc} \n")

# Write the results into file dt_balanced_acc.txt
try:
    f = open('../output/dt_balanced_acc.txt', 'w')
    f.write(f"dt_balanced_acc_train = {train_acc} \n")
    f.write(f"dt_balanced_acc_test = {test_acc} \n")
    f.close()
except Exception as e:
    print(e)

max_depths = []  # x-axis
balanced_acc_scores = []  # y-axis
best_max_depth = 0
best_score = 0
best_dtc = None
try:
    f = open('../output/dt_balanced_acc_scores.txt', 'w')
    # Test decision tree classifiers with max depth from 1 to 10
    for i in range(1, 11):
        # Create Decision Tree Classifier with max depth i using entropy
        dtc = DecisionTreeClassifier(criterion="entropy", max_depth=i, random_state=random_seed)
        # Train Decision Tree with training data
        dtc.fit(x_train, y_train)
        # Predict class of test data
        test_predictions = dtc.predict(x_test)
        # Get balanced accuracy score
        test_acc = balanced_accuracy_score(y_test, test_predictions)
        print(f"dt_entropy_max_depth_{i}_balanced_acc = {test_acc} \n")
        # Add max depth and balanced accuracy values for plotting
        max_depths.append(i)
        balanced_acc_scores.append(test_acc)
        f.write(f"dt_entropy_max_depth_{i}_balanced_acc = {test_acc} \n")
        # Check which Decision Tree has the best accuracy at what max depth
        if test_acc > best_score:
            best_score = test_acc
            best_max_depth = i
            best_dtc = dtc
    f.close()
except Exception as e:
    print(e)

# Plot balanced accuracy scores for all decision trees and save into PNG image
plt.figure(1)
plt.plot(max_depths, balanced_acc_scores, marker="o")
plt.title("DT Balanced Accuracy Scores")
plt.xlabel("Max Depth")
plt.ylabel("Balanced Accuracy Scores")
plt.savefig("../output/dt_balanced_acc_scores.png")

# Draw Decision Tree with the best balanced accuracy score
plt.figure(2)
tree.plot_tree(best_dtc)
plt.savefig(f"../output/dt_entropy_max_depth_{best_max_depth}.png", format='png', bbox_inches="tight")



