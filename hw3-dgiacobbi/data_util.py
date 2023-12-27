"""Data utility functions for CPSC 322 HW-3. 

Basic functions for preprocessing and visualization data sets. 

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""


from data_table import DataTable, DataRow
import matplotlib.pyplot as plt



def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    distinct_list = []

    # Add distinct values into list
    for row in table:

        # Grab each value from table row and store in val
        val_row = row.values(column)
        val = val_row[0]

        # Check if value has not been found yet
        if val not in distinct_list:
            distinct_list.append(val)
    
    return distinct_list


def remove_missing(table, columns):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    # Track which rows do not have empty values
    keep_idx = []
    idx_count = 0

    # Traverse current table and check each row
    for row in table:
        
        # Add index to rows that have all values
        if "" not in row.values(columns):
            keep_idx.append(idx_count)
        idx_count += 1

    return table.rows(keep_idx)


def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """
    # Track which rows are duplicates and unique
    unique_list = []
    duplicate_list = []

    duplicate_table = DataTable(table.columns())

    # Traverse table and check if current row is a duplicate
    for row in table:

        # Add row to unique_list if not already in the unique list, otherwise add to duplicate table and list
        if row.values() not in unique_list:
            unique_list.append(row.values())
        elif row.values() not in duplicate_list:
            duplicate_list.append(row.values())
            duplicate_table.append(row.values())

    return duplicate_table

                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    # Track unique rows and new DataTable to add rows
    unique_list = []
    unique_table = DataTable(table.columns())

    # Traverse table for unique rows and add to new DataTable
    for row in table:

        # Check if row has been added yet to unique_table
        if row.values() not in unique_list:
            unique_list.append(row.values())
            unique_table.append(row.values())
    
    return unique_table



def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on
    """
    # List for DataTables and list for every unique partition
    partition_list = []
    unique_partitions = []

    # Traverse every row in table provided
    for row in table:

        # If current row is unique, add to list and add new DataTable to list
        if row.values(columns) not in unique_partitions:

            # Add new table to partition_list of tables with current row
            new_table = DataTable(row.columns())
            new_table.append(row.values())
            partition_list.append(new_table)

            # Update unique partition check
            unique_partitions.append(row.values(columns))
        
        else:
            # Traverse tables in partition list and check where to add duplicate partition row
            for table in partition_list:
                if table[0].values(columns) == row.values(columns):
                    table.append(row.values())
    
    return partition_list



def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """
    # Remove missing values from column of current table
    stat_table = remove_missing(table, [column])
    stat_list = []
    
    # Traverse every row in table and append value from specified column in stat_list
    for row in stat_table:
        stat_list.append(row[column])

    # Compute summary stat function on appended list
    return function(stat_list)



def replace_missing(table, column, partition_columns, function): 
    """Replace missing values in a given table's column using the provided
     function over similar instances, where similar instances are
     those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """
    # Track the index of the current row through loop and new table
    idx = 0
    update_table = table.copy()

    # Traverse table and check each row for missing value to replace
    for row in update_table:
        
        # If row has missing value in specified column, perform partition and summary stat replacement
        if "" == row[column]:

            # Partition table by provided columns and find table with same values as current row in partition_columns
            partition_list = partition(table, partition_columns)

            for part in partition_list:
                # If a match is found perform summary stat and current table partition
                if part[0].values(partition_columns) == row.values(partition_columns):

                    # Get summary stat and update row
                    new_val = summary_stat(part, column, function)
                    update_table.update(idx, column, new_val)
        idx += 1 

    return update_table   


def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """
    # Create list to hold stat calculations and list of partitioned tables
    stat_list = []
    partition_list = partition(table, partition_column)

    # Visit each partitioned table and add its summary stat to the list
    for part in partition_list:
        new_stat = summary_stat(part, stat_column, function)
        stat_list.append(new_stat)
    
    # List of groups, then list of stats
    return distinct_values(table, partition_column), stat_list


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """
    # Create list to hold stat calculations and list of partitioned tables
    frequency_list = []
    partition_list = partition(table, partition_column)

    # Visit each partitioned table and add its summary stat to the list
    for part in partition_list:
        new_frequency = part.row_count()
        frequency_list.append(new_frequency)
    
    # List of groups, then list of stats
    return distinct_values(table, partition_column), frequency_list


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # reset figure
    plt.figure()
    # dummy y values
    yvalues = [1] * len(xvalues)
    # create an x-axis grid
    plt.grid(axis='x', color='0.85', zorder=0)
    # create the dot chart (with pcts)
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=16, zorder=3)
    # get rid of the y axis
    plt.gca().get_yaxis().set_visible(False)
    # assign the axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Reset figure
    plt.figure()

    # Create the bar chart ( with pcts )
    plt.pie(values, labels=labels, autopct ='%1.1f%%')
    
    # Title and labels
    plt.title(title)

    # Save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    # Close the plot
    plt.close()


def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_labels: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Reset figure
    plt.figure()

    # Define x values
    xs = [i for i in range(len(bar_names))]

    # Create a y-axis grid
    plt.grid(axis='y', color='0.85', zorder=0)

    # Create the bar chart (.45 relative bar width)
    plt.bar(xs, bar_values, width=0.45, align='center', zorder=3)

    # Define labels for x values
    plt.xticks(xs, bar_names)

    # x and y axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Title
    plt.title(title)
    
    # Save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    # Close the plot
    plt.close()

    
def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Reset figure
    plt.figure()

    # Add a grid
    plt.grid(color ='0.85 ', zorder=0)

    # Create the scatter plot
    plt.plot(xvalues, yvalues, color='b', marker ='.', alpha=0.2, markersize=16, linestyle='', zorder=3)

    # x and y labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Title
    plt.title(title)
    
    # Save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    # Close the plot
    plt.close()


def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Reset figure
    plt.figure()

    # Set a y- axis only grid
    plt.grid(axis='y', color ='0.85 ', zorder=0)

    # Create the box plot
    plt.boxplot(distributions, zorder=3)

    # Set the x- axis distribution names
    plt.xticks([i+1 for i in range(len(labels))], labels)

    # Set the x and y axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Title
    plt.title(title)

    # Save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    # Close the plot
    plt.close()