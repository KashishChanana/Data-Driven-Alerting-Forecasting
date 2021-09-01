import pandas as pd
from utils.clean import *


def load_data():
    """
    Loads datasets from json files and converts them to dataframe
    :return: df: Concatenated dataframe after cleaning
    """
    df_seller_pre = pd.read_json('./data/sellerlist_pre.json')
    df_seller_active = pd.read_json('./data/sellerlist_active.json')
    df_seller_post = pd.read_json('./data/sellerlist_post.json')
    df_characteristics = pd.read_json("./data/sellerlist_characteristics.json")

    # Alot of sellers corresponded to  multiple rows; need to remove duplicates and keep only one row per user with latest data
    df_characteristics = df_characteristics.drop_duplicates(subset='USER_ID', keep="last")
    df_characteristics.reset_index(inplace=True)
    df_characteristics.drop(columns=["index"], inplace=True)

    print("Loading data ... \n \n")
    print(df_characteristics.head(2))
    print("\n")
    print(df_seller_pre.head(2))
    print("\n")
    print(df_seller_active.head(2))
    print("\n")
    print(df_seller_post.head(2))
    print("\n")

    print("Shapes \n \n")
    print("PRE", df_seller_pre.shape)
    print("POST", df_seller_post.shape)
    print("ACTIVE", df_seller_active.shape)

    df = clean(df_seller_pre, df_seller_post, df_seller_active, df_characteristics)
    return df
