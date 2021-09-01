import os
import pandas as pd


def run_query(query, columns=None):
    """
    Runs query on the database and creates dataframe.

    Parameters
    ----------------
    :param query: Clickhouse query
    :param columns: list of columns of the database

    Returns
    ----------------
    Dataframe of retrieved data.
    """
    print("\n Fetching data ....\n \n")
    url = "curl -u @username:password -k @host " \
          "-d "
    url = url + " '" + query + "'"
    result = os.popen(url).read()
    result = [line.split('\t') for line in result.splitlines()]
    result = pd.DataFrame(result, columns=columns)

    return result
