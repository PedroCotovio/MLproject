# Imports
import nltk
from nltk.stem import WordNetLemmatizer as lemmatizer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# All functions used in the Exploratory Data Analysis of the ML project

# Load data function
def load_data(fname):
    """Load CSV file with any number of consecutive features, starting in column 0, where last column is tha class"""
    df = pd.read_csv(fname)
    nc = df.shape[1]  # number of columns
    matrix = df.values  # Convert dataframe to darray
    table_X = matrix[:, 0:nc - 1]  # get featuresD
    table_X = np.delete(table_X, 21, axis=1)  # remove petID
    table_X = np.delete(table_X, 18, axis=1)  # remove RescuerID
    table_y = matrix[:, nc - 1]  # get class (last columns)
    features = df.columns.values[0:nc - 1]  # get features names
    target = df.columns.values[nc - 1]  # get target name
    return df, table_X, table_y, features, target


# Pie Chart
def make_pie(x):
    """
    Plot dataset in Pie Chart

    :param x: array, dataset to plot
    """
    # get counts
    counts = []
    for i in set(x):
        counts.append(np.sum(x == i))
    # get explode
    explode = []
    for i in counts:
        if (i / sum(counts) * 100) < 5:
            explode.append(0.5)
        else:
            explode.append(0)

    # Draw pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, explode=explode, labels=set(x), autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    plt.show()

# Complex Bar plot
def prepare_plot_dict(df, col, main_count):
    """
    Return dictionary with column counts

    :param df: dataset, dataset to plot
    :param col: str, column to plot
    :param main_count: dict, comparison column counts

    From https://www.kaggle.com/artgor/exploration-of-data-step-by-step/notebook
    """
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0

    return plot_dict


def make_count_plot(df, x, hue='AdoptionSpeed', title='', fig=True):
    """
    Return bar chart with comparisons by percentage

    :param df: dataset, dataset to plot
    :param x: str, column to plot
    :param hue: str, comparison column
    :param title: str, Graph title
    :param fig: Bool, if figure size should be fixed

    From https://www.kaggle.com/artgor/exploration-of-data-step-by-step/notebook
    """

    main_count = df[hue].value_counts(normalize=True).sort_index()
    if fig is True:
        plt.figure(figsize=(18, 8));
    g = sns.countplot(x=x, data=df, hue=hue);
    plt.title = 'AdoptionSpeed and ' + str(title)
    ax = g.axes

    plot_dict = prepare_plot_dict(df, x, main_count)

    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0,
                    xytext=(0, 10),
                    textcoords='offset points')


# Bar Chart
def make_bar_chart(x):
    """
    Return Bar Chart

    :param x: array, dataset to plot

    """
    plt.figure(figsize=(14, 6));
    g = sns.countplot(x)
    ax = g.axes
    for p in ax.patches:
        ax.annotate(f"{p.get_height() * 100 / x.shape[0]:.2f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
                    textcoords='offset points')


# Check Pet breeds
def breeds_check(ls, off):
    """
    Check if a string represents a known animal breed

    :param ls: list, strings to check
    :param off: list, official breeds
    :return: list, strings that are not breeds
    """
    not_breeds = []
    for x in range(len(ls)):
        temp = ls[x].split()
        count = 0
        for word in temp:
            if word in ['DOG', 'CAT', 'BROWN'] or 'HAIR' in word:
                continue
            for breed in off:
                if word in breed:
                    count += 1
        if count == 0:
            not_breeds.append(ls[x])
    return not_breeds


# Check if is float
def is_number(string):
    """
    Return whether the string can be interpreted as a float.

    :param string: str, string to check for float
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


# Get range between two letters
def ab_range(a, b, df, x):
    """
    Create an alphabetical range between to given letters
    and slice dataset for all values from a certain column
    that start with the letters in that range

    :param a: str, first letter
    :param b: str, last letter
    :param df: dataset, original
    :param x: str, column in dataset to slice by
    :return: dataset, sliced
    """
    letters = [chr(i) for i in range(ord(a), ord(b) + 1)]
    res = []
    for i, name in enumerate(df[x]):
        if is_number(name) is True or name.isdigit():
            continue
        elif df.loc[i, x][0].lower() in letters:
            res.append(i)
    # [idx for idx in range(len(df['Name'])) if df.loc[idx, 'Name'][0].lower() in letters and df.loc[idx, 'namecat'] == 2]
    return res


# Feature Extractor Names
def name_features(name):
    """
    Count letters in string

    :param name: str, string to count from
    :return: dict, all letters and respective counts
    """
    features = {}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        # features["has({})".format(letter)] = (letter in name.lower())
        features["count({})".format(letter)] = name.lower().count(letter)
    return features


# Feature Extractor Description
def document_features(document, stop, tagset):
    """
    Count tags in multi word string

    :param document: str, multi word string to evaluate
    :param stop: list, words to ignore (stopwords)
    :param tagset: list, all possible tags
    :return: dict, all tags and respective counts
    """
    # Get Tokens & remove stopwords
    tokens = [lemmatizer.lemmatize(lemmatizer, w.lower()) for w in nltk.word_tokenize(str(document)) if w not in stop]
    # Tag tokens
    tokens = nltk.pos_tag(tokens)
    # Get tags
    tags = [tag[1] for tag in tokens]
    # Create Features
    features = {}
    for tag in tagset['Label']:
        features["count({})".format(tag)] = tags.count(tag)
    return features


# Capture prints
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import sys


class Capturing(list):
    """
    Capture output of print line by line into list as string
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# Classification Check
def pred_check(pred, n):
    """
    Check if all classes are being classified

    :param pred: list, predictions made by classifier
    :param n: int, number of possible classes
    :return: bool, are all classes being classified ?
    """
    if len(set(pred)) == n:
        return True
    else:
        return False


# Encode Features

def int_encode_class(vect):
    """
    Encode classes as ints

    :param vect: list, values to encode
    :return: list, encoded values
    """
    enc = LabelEncoder()
    label_encoder = enc.fit(vect)
    integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(vect)
    return t


def int_encode_feature(vect):
    return int_encode_class(vect)

# Unsupervised comparisons
def comp_val(tar, res):
    """
    Compare unsupervised learning results against known labels
    :param tar: list, known labels
    :param res: list, results
    :return: dataframe, confusion matrix
    """
    cl = set(tar)
    result = [{x:0 for x in cl} for i in range(max(res)+1)]
    for i, c in enumerate(res):
        result[c][tar[i]] += 1
    result = pd.DataFrame(result)
    return result
