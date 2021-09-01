import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

sys.path.append('../pricingml/utils')
from mapping import *

plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['figure.figsize'] = 20, 10


feecode_map, countryid_map, substier_map = initialize_mapping()

def plot(x, y, title, x_label=None, y_label=None):
    """
    Plots a simple plot.

    Parameters
    ----------------
    :param x : x axis points
    :param y : y axis points
    :param x_label : label associated with x axis
    :param y_label : label associated with y axis

    Returns
    ----------------
    A matplotlib figure.
    """
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    # plt.xlabel = x_label
    # plt.ylabel = y_label
    plt.show()


def comparison_plot(df_train, df_test, predictions, predictions_lower, predictions_upper, model_name, fee_code,
                    listing_site_id, all_trends=True):
    """
    Plots observed trends and predicted trends.

    Parameters
    ----------------
    :param df_train : training dataframe
    :param df_test : testing dataframe
    :param predictions : predictions obtained from the model model_name
    :param predictions_lower : lower bound on predictions with 80 percent confidence
    :param predictions_upper : upper bound on predictions with 80 percent confidence
    :param model_name : name of the model being used
    :param fee_code: code of fee_type being analyzed
    :param listing_site_id : listing site id on eBay of the site being analyzed

    Returns
    ----------------
    A matplotlib figure.
    """

    fee_column = 'total_fee_' + fee_code

    plt.figure()
    plt.plot(df_test["datetime"], df_test[fee_column].values, color='red', label="Testing data")
    plt.plot(df_test["datetime"], predictions, label="Predictions using " + model_name, color="green")

    if all_trends:
        plt.plot(df_train["datetime"], df_train[fee_column].values, label="Training data")

    if predictions_upper is not None and predictions_lower is not None and (
            model_name == 'Prophet' or model_name == 'Prophet-Multi'):
        plt.fill_between(df_test["datetime"], predictions_lower, predictions_upper, color='cyan',
                         label="Within 80 percent confidence")

    plt.title("Daily Trends of " + feecode_map[int(fee_code)] + " on site " + countryid_map[
        listing_site_id] + " using "
              + model_name)

    plt.xlabel("datetime")
    plt.ylabel("Fee" + fee_code)
    plt.legend()
    plt.show()


def bar_plot(evaluationsDict):
    """
    Plots a bar plot using the key, values of the dictionary.

    Parameters
    ----------------
    :param evaluationsDict : key-value pairs of models with RMSE values on test set

    Returns
    ----------------
    A bar plot.
    """
    plt.figure()
    evaluationsDict = dict(sorted(evaluationsDict.items(), key=lambda item: item[1]))

    plt.bar(evaluationsDict.keys(), evaluationsDict.values())

    plt.title("Comparison of RMSE across models")
    plt.ylabel("Root Mean Squared Error")
    plt.xlabel("Model Name")
    plt.show()
