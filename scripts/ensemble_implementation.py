# STUDENT_ID: 100276935
# Created on: 2022-11-14
# Last update: 2022-12-13, [last modification to your code]
# Description:  Creates and calculates the balanced accuracy of an ensemble of Decision Tree Classifiers with different
#               criterion and max depths. Results are written into file ensemble_balanced_acc.txt. An array of decision
#               trees with different parameters are created which will be used for an ensemble. The ensemble uses
#               majority vote to predict the test data.

import pandas
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score


import helper_functions as hf

random_seed = 1

file_path = '../data/breast-cancer.data'
hf.check_data(file_path)

# Loads data from csv into variable
csv_data = pandas.read_csv('../data/breast-cancer.data', header=None)

# Convert categorical (string) attribute values to numerical (integer)
data, unique_attr = hf.category_to_numerical(csv_data)
# Split the data into training and test sets
x_train, y_train, x_test, y_test = hf.split_data(data, 0.25, random_seed)


def create_decision_tree_classifiers():
    """
        Function that creates an array of decision tree classifiers. Merely for coursework parameters
        :return: Array of decision tree classifiers
    """
    dt_arr = []
    for i in range(1, 6):
        selection_type = ""
        depth = 0
        if i % 2 == 0:
            selection_type = "entropy"
        else:
            selection_type = "gini"
        if i > 1:
            depth = i*5 + 5
        else:
            depth = 5
        decision_tree_classifier = DecisionTreeClassifier(criterion=selection_type, max_depth=depth,
                                                          random_state=random_seed)
        dt_arr.append(decision_tree_classifier)
    return dt_arr


def ensemble_majority_vote_prediction(classifiers):
    """
        Function that creates an ensemble by using all the classifiers given and returns the prediction based on
        majority vote of all predictions by each classifier
        :param classifiers: Array of classifiers
        :return: Array of predictions of each instance
    """
    predictions = []
    # Loop through each decision tree to train using training data and predict the class on test data. Append each
    # array of predictions into the predictions list. The result will be a list of decision tree prediction arrays.
    for classifier in classifiers:
        classifier.fit(x_train, y_train)
        predictions.append(classifier.predict(x_test))
    # List to store the overall prediction of all classifiers
    majority_vote_prediction = []
    # Loop through each item in the first array to get index of class predictions for each decision tree prediction
    for i in range(len(predictions[0])):
        # Array to store class values of each decision tree at index i
        class_values = []
        # Store the prediction of each decision tree at index i in an array
        for j in range(len(predictions)):
            class_values.append(predictions[j][i])
        # Check which class occurs the most frequent in the class_values array and add it to the list.
        majority_vote_prediction.append(hf.most_common(class_values))
    return majority_vote_prediction


# Create an array of decision trees with different parameters
decision_tree_classifiers = create_decision_tree_classifiers()
# To store predictions of each decision tree
ensemble_predictions = ensemble_majority_vote_prediction(decision_tree_classifiers)

# decision_tress = [("DT0", decision_tree_classifiers[0]), ("DT1", decision_tree_classifiers[1]),
#                   ("DT2", decision_tree_classifiers[2]), ("DT3", decision_tree_classifiers[3]),
#                   ("DT4", decision_tree_classifiers[4])]
#
# majority_vote = VotingClassifier(estimators=decision_tress, voting='hard')
# majority_vote.fit(X=x_train, y=y_train)
# ensemble_predictions = majority_vote.predict(x_test)

# Get the overall prediction from ensemble
balanced_accuracy = balanced_accuracy_score(y_test, ensemble_predictions)

print(f"ensemble_balanced_acc = {balanced_accuracy}")

try:
    f = open('../output/ensemble_balanced_acc.txt', 'w')
    f.write(f"ensemble_balanced_acc = {balanced_accuracy} \n")
    f.close()
except Exception as e:
    print(e)

