"""Data visualization testing for CPSC 322 HW-3. 

Basic test cases for visualization data sets. 

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

from random import randint
from data_table import *
from data_util import *

def test_dot_chart():
    x_values = [1, 1, 1, 1, 1, 1, 1, 1.1, 1.2, 1.2, 1.3, 1.4, 1.5, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
    x_label = "Example X Label"
    title = "Example Dot Chart"
    dot_chart(x_values, x_label, title)

def test_pie_chart():
    categories = ['a', 'b', 'c', 'd']
    counts = [100 , 200 , 400 , 300]
    title = "Example Pie Chart"
    pie_chart(counts, categories, title)

def test_bar_chart():
    bar_values = [1000, 300, 750, 960, 410, 500]
    bar_names = ['fe', 'fi', 'fo', 'fum', 'jim', 'bob']
    x_label = "Example X Label"
    y_label = "Example Y Label"
    title = "Example Bar Chart"
    bar_chart(bar_values, bar_names, x_label, y_label, title)

def test_scatter_plot():
    x_values = [randint(0, round(i/2)) for i in range(1, 100)]
    y_values = [randint(0, round(i/2)) for i in range(1, 100)]
    x_label = "Example X Label"
    y_label = "Example Y Label"
    title = "Example Scatter Plot"
    scatter_plot(x_values, y_values, x_label, y_label, title)

def test_box_plot():
    xs1 = [randint(0, round(i/2)) for i in range(1, 100)]
    xs2 = [randint(0, round(i/2)) for i in range(1, 100)]
    xs3 = [randint(0, round(i/2)) for i in range(1, 100)]
    distributions = [xs1, xs2, xs3]
    labels = ['xs1', 'xs2', 'xs3']
    x_label = "Example X Label"
    y_label = "Example Y Label"
    title = "Example Box Plot"
    box_plot(distributions, labels, x_label, y_label, title)

test_dot_chart()
test_pie_chart()
test_bar_chart()
test_scatter_plot()
test_box_plot()