# STUDENT_ID: 100276935
# Created on: 2022-11-13
# Last update: 2022-12-13
# Description: Contains functions that will be used for operations that may need to be used multiple times

from sklearn.model_selection import ShuffleSplit
import itertools as it


def input_contingency_table(yes_pos=None, yes_neg=None, no_pos=None, no_neg=None):
    """
        Function that creates a Contingency Table using manual input or arguments. Only works for calculating the
        first split
        :param yes_pos: Number of instances with attribute Yes and class Positive
        :param yes_neg: Number of instances with attribute Yes and class Negative
        :param no_pos: Number of instances with attribute No and class Positive
        :param no_neg: Number of instances with attribute No and class Negative
        :return: Contingency table as an array, total instances with Positive, total instances with Negative
    """
    # Contingency table =
    # [ [ attribute_value_1_positive_count, attribute_value_1_negative_count ],
    #   [ attribute_value_2_positive_count, attribute_value_2_negative_count ] ]
    table = [[0, 0],
             [0, 0]]
    if yes_pos is None or yes_neg is None or no_pos is None or no_neg is None:
        print("One or more arguments not given. Manual input required.")
        while True:
            yes_pos = input("Enter the number of instances with Yes AND Positive Values: ")
            yes_neg = input("Enter the number of instances with Yes AND Negative Values: ")
            no_pos = input("Enter the number of instances with No AND Positive Values: ")
            no_neg = input("Enter the number of instances with No AND Negative Values: ")
            try:
                yes_pos, yes_neg, no_pos, no_neg = int(yes_pos), int(yes_neg), int(no_pos), int(no_neg)
                # if not a positive int print message and ask for input again
                if yes_pos < 0 or yes_neg < 0 or no_pos < 0 or no_neg < 0:
                    print("Sorry, all inputs must be a positive integer. Try again")
                    continue
                break
            except ValueError:
                print("One or more inputs are not an int! Try again.")

    total_positive = yes_pos + no_pos
    total_negative = yes_neg + no_neg
    table = [[yes_pos, yes_neg], [no_pos, no_neg]]

    return table, total_positive, total_negative


def check_data(file_path):
    """
        Function that checks validity of data in file and removes any invalid rows of data
        :param file_path: Path of file
    """
    try:
        # Reads the file and saves all lines
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()

        # Checks if each line is valid before writing to same file. If line is not valid, do not write
        f = open(file_path, 'w')
        for line in lines:
            if "?" not in line or ",," not in line:
                f.write(line.lower())
        f.close()
    except Exception as e:
        print(e)


def category_to_numerical(data_frame):
    """
        Function that converts categorical (String) values of data frame into integer
        :param data_frame: Data Frame
        :return: Data Frame, list of attributes for each numerical value
    """
    shape = data_frame.shape
    # Use the first instance to check the data type of each attribute
    instance = data_frame.loc[0]
    attributes_list = []
    # Loop through the columns
    for i in range(0, shape[1]):
        # Check if the value data type is string, if yes, replace all as a numerical value
        if isinstance(instance[i], str):
            # Get all unique values of the column
            arr = data_frame[i].unique()
            # Sort the values
            arr.sort()
            # Append the unique values to an array to know which number belongs to which value
            attributes_list.append(arr)
            # Transform categorical values into numerical representation
            data_frame[i].replace(arr, [e for e in range(0, len(arr))], inplace=True)
    return data_frame, attributes_list


# Splits (data) into train and test data and labels by size (test_size) with random state (random_seed)
def split_data(data, test_size, random_seed):
    """
        Function that splits data into training and test data
        :param data: Data
        :param test_size: Percentage size of test data (0 < test_size < 1)
        :param random_seed: Random Seed initialiser
        :return: , list of attributes for each numerical value
    """
    class_list = data[0]
    data.drop(columns=0, inplace=True)
    y = class_list.values
    x = data.values
    split_function = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    instance_index_list = list(split_function.split(x, y))
    train_index_set = instance_index_list[0][0]
    test_index_set = instance_index_list[0][1]

    x_train = x[train_index_set, :]
    y_train = y[train_index_set]
    x_test = x[test_index_set, :]
    y_test = y[test_index_set]

    return x_train, y_train, x_test, y_test


def most_common(arr):
    """
        Function that returns element with the highest number of occurrences in list
        :param arr: array
        :return: Element with highest occurences
    """
    return max(set(arr), key=arr.count)


# https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
def get_combinations(param_dict):
    """
        Function that returns all combinations of values in dictionary of list
        :param param_dict: Dictionary of parameter lists
        :return: array of key-value parameter combination dictionaries
    """
    keys, values = zip(*param_dict.items())
    combinations = [dict(zip(keys, v)) for v in it.product(*values)]
    return combinations

