# STUDENT_ID: 100276935
# Created on: 2022-11-04
# Last update: 2022-12-13, [last modification to your code]
# Description:  Contains methods to calculate the Information Gain, Gini Index, and Chi-Squared Statistic of the
#               Headache attribute. Results are written to headache_splitting_diagnosis.txt.

import pandas
from math import log
import helper_functions as hf

random_seed = 1

# Created a diagnosis.data file for creating the headache contingency table.
file_path = '../data/diagnosis.data'
# Checks data and removes any lines containing ? or empty values
hf.check_data(file_path)
# Loads data from csv into variable as a Data Frame
data = pandas.read_csv('../data/diagnosis.data', header=None)


def get_entropy(frac_1, frac_2):
    """
        Function that calculates entropy value for Information Gain
        :param frac_1: Fraction value as Double
        :param frac_2: Fraction value as Double
        :return: Entropy value
    """
    entropy = -(frac_1 * log(frac_1, 2) + frac_2 * log(frac_2, 2))
    return entropy


# Calculates the impurity measure of a node
def get_impurity(frac_1, frac_2):
    """
        Function that calculates impurity measure for Gini Index
        :param frac_1: Fraction value as Double
        :param frac_2: Fraction value as Double
        :return: Impurity measure value
    """
    impurity = 1 - frac_1**2 - frac_2**2
    return impurity


def create_headache_contingency_table(csv_data):
    """
        Function that creates contingency table for headache
        :param csv_data: Data from CSV
        :return: Impurity measure value
    """
    # Contingency table =
    # [(attribute_value_1) [ positive_count, negative_count ],
    #  (attribute_value_2) [ positive_count, negative_count ]]
    table = [[0, 0],
             [0, 0]]
    total_pos = 0
    total_neg = 0

    for i in range(0, len(csv_data)):
        instance = csv_data.loc[i]  # Read instance at row i
        headache_value = instance[0]  # Get the headache value from column 0
        diagnosis_value = instance[3]  # Get the diagnosis value from column 3
        # Checks the attribute and class values
        if headache_value == "yes" and diagnosis_value == "positive":
            table[0][0] += 1
            total_pos += 1
        if headache_value == "yes" and diagnosis_value == "negative":
            table[0][1] += 1
            total_neg += 1
        if headache_value == "no" and diagnosis_value == "positive":
            table[1][0] += 1
            total_pos += 1
        if headache_value == "no" and diagnosis_value == "negative":
            table[1][1] += 1
            total_neg += 1
    return table, total_pos, total_neg


# Uses data frame from CSV to build headache contingency table
contingency_table, total_positive, total_negative = create_headache_contingency_table(data)
# Uses arguments or manual input to build headache contingency table
# contingency_table, total_positive, total_negative = hf.input_contingency_table(4, 2, 3, 5)
total_count = total_positive + total_negative
# Get entropy and impurity for root node
parent_entropy = get_entropy(total_positive/total_count, total_negative/total_count)
parent_impurity = get_impurity(total_positive/total_count, total_negative/total_count)


def get_information_gain(table):
    """
        Function that calculates Information Gain of an attribute
        :param table: table of values as a 2D array
        :return: Information Gain value
    """
    yes_pos = table[0][0]
    yes_neg = table[0][1]
    no_pos = table[1][0]
    no_neg = table[1][1]
    yes_total = yes_pos + yes_neg
    no_total = no_pos + no_neg
    total = yes_total + no_total
    yes_entropy = get_entropy(yes_pos/yes_total, yes_neg/yes_total)
    no_entropy = get_entropy(no_pos/no_total, no_neg/no_total)
    info_gain = parent_entropy - (yes_total/total)*yes_entropy - (no_total/total)*no_entropy
    return info_gain


def get_gini(table):
    """
        Function that calculates Gini Index of an attribute
        :param table: table of values as a 2D array
        :return: Gini Index value
    """
    yes_pos = table[0][0]
    yes_neg = table[0][1]
    no_pos = table[1][0]
    no_neg = table[1][1]
    yes_total = yes_pos + yes_neg
    no_total = no_pos + no_neg
    total = yes_total + no_total

    yes_measure = get_impurity(yes_pos/yes_total, yes_neg/yes_total)
    no_measure = get_impurity(no_pos/no_total, no_neg/no_total)
    gini_index = parent_impurity - (yes_total/total)*yes_measure - (no_total/total)*no_measure
    return yes_measure, no_measure, gini_index


def get_chi_squared(table):
    """
        Function that calculates Chi Squared Statistic of an attribute
        :param table: table of values as a 2D array
        :return: Chi-Squared Statistic value
    """
    yes_pos = table[0][0]
    yes_neg = table[0][1]
    no_pos = table[1][0]
    no_neg = table[1][1]
    yes_total = yes_pos + yes_neg
    no_total = no_pos + no_neg

    expected_yes_pos = yes_total * (total_positive/total_count)
    expected_yes_neg = yes_total * (total_negative/total_count)
    expected_no_pos = no_total * (total_positive/total_count)
    expected_no_neg = no_total * (total_negative/total_count)

    yes_chi = ((yes_pos - expected_yes_pos)**2)/expected_yes_pos + ((yes_neg - expected_yes_neg)**2)/expected_yes_neg
    no_chi = ((no_pos - expected_no_pos)**2)/expected_no_pos + ((no_neg - expected_no_neg)**2)/expected_no_neg
    chi_squared = yes_chi + no_chi
    return chi_squared


information_gain_result = get_information_gain(contingency_table)
yes_impurity, no_impurity, gini_index_result = get_gini(contingency_table)
chi_squared_result = get_chi_squared(contingency_table)
print("measure_headache_information_gain =", information_gain_result)
print("measure_headache_yes_gini_impurity_measure =", yes_impurity)
print("measure_headache_no_gini_impurity_measure =", no_impurity)
print("measure_headache_gini_index =", gini_index_result)
print("measure_headache_chi_squared =", chi_squared_result)

# Write to file
try:
    f = open('../output/headache_splitting_diagnosis.txt', 'w')
    f.write(f"measure_headache_information_gain = {information_gain_result} \n")
    f.write(f"measure_headache_yes_gini_impurity_measure = {yes_impurity} \n")
    f.write(f"measure_headache_no_gini_impurity_measure = {no_impurity} \n")
    f.write(f"measure_headache_gini_index = {gini_index_result} \n")
    f.write(f"measure_headache_chi_squared = {chi_squared_result} \n")
    f.close()
except:
    print("Oops! Something went wrong")
