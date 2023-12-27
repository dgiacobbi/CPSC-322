"""Machine learning algorithm implementations.

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import DataTable, DataRow


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """
    # Create empty dictionary to hold k key-value pairs
    row_dist = dict()

    # Traverse table to calculate each row distance 
    for row in table:
        distance = 0

        # Calculate numerical column square differences
        for num_label in numerical_columns:
            cur_dif_sq = (row[num_label] - instance[num_label]) ** 2
            distance = distance + cur_dif_sq

        # Calculate nominal column square diffferences
        for nom_label in nominal_columns:
            if instance[nom_label] != row[nom_label]:
                distance = distance + 1
        
        # Check if distance has any row values currently: append or create new list
        if distance not in row_dist.keys():
            row_dist[distance] = [row]
        else:
            row_dist[distance] += [row]

    # Sort dictionary items and create return dictionary for k number of values
    sorted_dist_items = sorted(row_dist.items())
    k_row_dist = dict()

    # Add first k elements of sorted dictionary to return dictionary
    for key, value in sorted_dist_items:
        k_row_dist[key] = value
        if len(k_row_dist) == k:
            break

    return k_row_dist

def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    # Create a dictionary of vote counts for each label
    label_counts = dict()

    # Add each vote to dictionary where label is key and value is vote count
    for row in instances:

        vote = row[labeled_column]

        if vote not in label_counts.keys():
            label_counts[vote] = 1
        else:
            label_counts[vote] = label_counts[vote] + 1

    # Traverse dictionary and append keys with max value to list
    majority_list = []
    for key, value in label_counts.items():
        if value == max(label_counts.values()):
            majority_list.append(key)

    return majority_list



def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    # Create a dictionary of aggregate scores for each label
    label_scores = dict()

    # Add each vote score to dictionary where label is key and value is aggregate score
    for i in range(len(instances)):

        vote = instances[i][labeled_column]

        if vote not in label_scores.keys():
            label_scores[vote] = scores[i]
        else:
            label_scores[vote] = label_scores[vote] + scores[i]

    # Traverse dictionary and append keys with max value to list
    majority_list = []
    for key, value in label_scores.items():
        if value == max(label_scores.values()):
            majority_list.append(key)

    return majority_list

