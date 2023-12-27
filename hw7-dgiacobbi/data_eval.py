"""Machine learning algorithm evaluation functions. 

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_learn import *
from data_util import *
from random import randint


#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(train, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)
    
    # Build decision tree
    predict_tree = tdidt(train, label_col, columns)
    attr_predict_tree = resolve_attribute_values(predict_tree, train)
    final_predict_tree = resolve_leaf_nodes(attr_predict_tree)

    # Perform decision tree on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform tdidt on current test instance
        predict_label, pred_prob = tdidt_predict(final_predict_tree, test_row)

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Stratify the table into k-folds
    stratify_table = stratify(table, label_col, k_folds)

    # Traverse each table section in k-folds and perform naive bayes on it as the test set
    for i in range(len(stratify_table)):

        # Combine the other tables to make the train set
        train_tables = []
        for j in range(len(stratify_table)):
            if j != i:
                train_tables.append(stratify_table[j])
        train = union_all(train_tables)

        # Build decision tree with train set
        predict_tree = tdidt(train, label_col, columns)
        attr_predict_tree = resolve_attribute_values(predict_tree, train)
        final_predict_tree = resolve_leaf_nodes(attr_predict_tree)

        # Perform decision tree on the table using train and test set
        for test_row in stratify_table[i]:

            # Get current label's index in confusion matrix
            cur_row_index = label_list.index(test_row[label_col])

            # Perform decision tree on current test instance
            predict_label, pred_prob = tdidt_predict(final_predict_tree, test_row)

            # Update confusion matrix row with prediction
            conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    # Partition the original table by class label
    label_partition = partition(table, [label_column])

    # Create a k-size list of empty data tables to distribute values
    stratify_list = [DataTable(table.columns()) for i in range(k)]

    # Traverse each class label partition
    for part in label_partition:

        # Bin Index to loop through stratify list
        bin_index = 0

        # Distribute rows with class label even across bins
        for row in part:

            stratify_list[bin_index].append(row.values())

            # Increment the bin index each time, looping back to 0 as needed
            bin_index += 1
            if bin_index == k:
                bin_index = 0

    return stratify_list


def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    # Check to make sure tables has at least one list
    if len(tables) == 0:
        raise ValueError

    # Create an empty data table to add values to
    union_table = DataTable(tables[0].columns())

    # Traverse tables list and add rows from each table
    for table in tables:

        # Check to make sure tables can be combined together
        if len(table.columns()) != len(union_table.columns()):
            raise ValueError

        for i in range(len(table.columns())):
            if table.columns()[i] != union_table.columns()[i]:
                raise ValueError

        # Add rows from current table to the union table
        for row in table:
            union_table.append(row.values())

    return union_table

def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(train, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Perform naive bayes on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform naive bayes on current test instance
        pred_labels, pred_prob = naive_bayes(train, test_row, label_col, continuous_cols, categorical_cols)

        # If more than one predict label in highest probability, select first
        predict_label = pred_labels[0]

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Stratify the table into k-folds
    stratify_table = stratify(table, label_col, k_folds)

    # Traverse each table section in k-folds and perform naive bayes on it as the test set
    for i in range(len(stratify_table)):

        # Combine the other tables to make the train set
        train_tables = []
        for j in range(len(stratify_table)):
            if j != i:
                train_tables.append(stratify_table[j])
        train = union_all(train_tables)

        # Perform naive bayes on the table using train and test set
        for test_row in stratify_table[i]:

            # Get current label's index in confusion matrix
            cur_row_index = label_list.index(test_row[label_col])

            # Perform naive bayes on current test instance
            pred_labels, pred_prob = naive_bayes(train, test_row, label_col, cont_cols, cat_cols)

            # If more than one predict label in highest probability, select first
            predict_label = pred_labels[0]

            # Update confusion matrix row with prediction
            conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Stratify the table into k-folds
    stratify_table = stratify(table, label_col, k_folds)

    # Traverse each table section in k-folds and perform naive bayes on it as the test set
    for i in range(len(stratify_table)):

        # Combine the other tables to make the train set
        train_tables = []
        for j in range(len(stratify_table)):
            if j != i:
                train_tables.append(stratify_table[j])
        train = union_all(train_tables)

        # Perform knn on the table using train and test set
        for test_row in stratify_table[i]:

            # Get current label's index in confusion matrix
            cur_row_index = label_list.index(test_row[label_col])

            # Perform knn on current test instance
            knn_dict = knn(train, test_row, k, num_cols, nom_cols)

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
            conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

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

    # Check if both are 0 (100% guess given zero instances):
    if total_predicted == 0 and true_positives == 0:
        return 1
    
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

