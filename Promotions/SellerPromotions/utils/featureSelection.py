import matplotlib.pyplot as plt
import seaborn as sns

def plot_multicollinearity(df):
    """
    Plots multicollinearity plot for the dataframe df.
    :param df: master Dataframe containing feature and target columns
    :return: None; displays plot
    """
    plt.figure(figsize=(9, 6))
    sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Multicollinearity", fontsize="x-large")
    plt.xticks(rotation=45, ha="right");


target = ["success"]

features = ["USER_ID", "SELLER_RESIDENCE_COUNTRY_ID", "IS_BUSINESS_SELLER",
            "IS_DIRECT_CHARITY_SELLER", "TOP_SELLER_LEVEL", "SUBS_TIER",
            "SELLER_SPS_LEVEL", "IS_P20_SELLER",
            "pre_dec_promo_discount", "pre_dec_base_charge", "pre_dec_list_items"]

def split_features_target(df, features_col=features, target_col=target):
    """
    Method for filtering features and target Series from master dataframe
    :param df: master Dataframe containing feature and target columns
    :param features_col: list with selected features column names
    :param target_col: list with selected target column name
    :return X: DataFrame with selected feature columns
    :return y: Series with selected target column

    """

    plot_multicollinearity(df)
    X = df[features_col]
    y = df[target_col]
    return X, y

