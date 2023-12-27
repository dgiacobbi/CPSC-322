"""Machine learning algorithm evaluation functions. 

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_learn import *
from data_util import *
from random import randint



def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    # Create an empty test set and copy the table to train set
    train_set = table.copy()
    test_set = DataTable(table.columns())

    # Given test set size, move rows from train set to test set
    for i in range(test_set_size):

        # Pick a random row from train set
        rand_index = randint(0, train_set.row_count() - 1)

        test_set.append(train_set[rand_index].values())
        del train_set[rand_index]

    return(train_set, test_set)


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(train, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Perform knn on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform knn on current test instance
        knn_dict = knn(train, test_row, k, numeric_cols, nominal_cols)

        # Create a list of all knn instances and corresponding scores
        knn_instances = []
        knn_scores = []

        for key, value in knn_dict.items():
            for instance in value:
                knn_instances.append(instance)         
                knn_scores.append(key)
        
        # Get the predicition from voting function
        maj_label_list = vote_fun(knn_instances, knn_scores, label_col)
        predict_label = maj_label_list[0]

        # Update confusion matrix row with prediction
        predict_idx = matrix_col.index(predict_label)
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix



def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    # Get the row index of label (take into account 'actual' column)
    col_list = confusion_matrix.columns()
    row_label_idx = col_list.index(label) - 1

    # Determine the true positive calculation
    true_positives = confusion_matrix[row_label_idx][label]

    # Calculate the true negatives for label
    true_negatives = 0

    for i in range(confusion_matrix.row_count()):

        # Sum values that are true negatives
        if row_label_idx != i:

            # Traverse each row and add values that are not actual counts or the label
            for attribute in col_list:        
                if attribute != label and attribute != 'actual':
                    true_negatives = true_negatives + confusion_matrix[i][attribute]

    # Calculate the number of predicted instances
    instance_counts = [0 for i in range(confusion_matrix.row_count())]

    for row in confusion_matrix:
        for label in col_list:
            if label != 'actual':
                instance_counts[col_list.index(label)-1] = instance_counts[col_list.index(label)-1] + row[label]

    instance_sum = sum(instance_counts)

    # Determine accuracy with variables found
    label_accuracy = (true_positives + true_negatives) / instance_sum
    return label_accuracy



def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    # Get the row index of label (take into account 'actual' column)
    col_list = confusion_matrix.columns()
    row_label_idx = col_list.index(label) - 1

    # Determine the true positive calculation
    true_positives = confusion_matrix[row_label_idx][label]

    # Determine the total predicted calculation
    total_predicted = 0
    for row in confusion_matrix:
        total_predicted = total_predicted + row[label]
    
    # Determine precision with variables
    return (true_positives) / (total_predicted)



def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    # Get the row index of label (take into account 'actual' column)
    col_list = confusion_matrix.columns()
    row_label_idx = col_list.index(label) - 1

    # Determine the true positive calculation
    true_positives = confusion_matrix[row_label_idx][label]

    # Determine the total actual labels
    total_positives = 0
    for i in range(1,len(col_list)):
        total_positives = total_positives + confusion_matrix[row_label_idx][col_list[i]]
    
    # Determine recall with variables
    return true_positives / total_positives