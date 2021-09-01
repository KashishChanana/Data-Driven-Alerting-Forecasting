import pandas as pd

def clean(df_seller_pre, df_seller_post, df_seller_active, df_characteristics):
    """
    Merges df_seller_pr, df_seller_post, df_seller_active, df_characteristics into master dataframe, df
    :param df_seller_pre: dataframe with figures before the promotion was launched
    :param df_seller_post: dataframe with figures after the promotion was over
    :param df_seller_active: dataframe with figures before the promotion was active
    :param df_characteristics: dataframe storing characteristics of the sellers
    :return: concatenated dataframe
    """
    df = df_characteristics
    df["dec_promo_discount"] = df_seller_active["sum(TOTAL_PROMO_DISCOUNT)"]
    df["dec_base_charge"] = df_seller_active["sum(TOTAL_BASE_CHARGE)"]
    df["dec_list_items"] = df_seller_active["uniqExact(ITEM_ID)"]

    pre_dec_list_items = []
    pre_dec_promo_discount = []
    pre_dec_base_charge = []

    for user in df["USER_ID"]:
        if user in list(df_seller_pre["USER_ID"]):
            idx = df_seller_pre[df_seller_pre["USER_ID"] == user].index.tolist()[0]
            pre_dec_list_items.append(df_seller_pre.iloc[idx]["uniqExact(ITEM_ID)"])
            pre_dec_promo_discount.append(df_seller_pre.iloc[idx]["sum(TOTAL_PROMO_DISCOUNT)"])
            pre_dec_base_charge.append(df_seller_pre.iloc[idx]["sum(TOTAL_BASE_CHARGE)"])

        else:
            pre_dec_list_items.append(0)
            pre_dec_promo_discount.append(0)
            pre_dec_base_charge.append(0)

    df["pre_dec_promo_discount"] = pre_dec_promo_discount
    df["pre_dec_base_charge"] = pre_dec_base_charge
    df["pre_dec_list_items"] = pre_dec_list_items

    post_dec_list_items = []
    post_dec_promo_discount = []
    post_dec_base_charge = []

    for user in df["USER_ID"]:
        if user in list(df_seller_post["USER_ID"]):
            idx = df_seller_post[df_seller_post["USER_ID"] == user].index.tolist()[0]
            post_dec_list_items.append(df_seller_post.iloc[idx]["uniqExact(ITEM_ID)"])
            post_dec_promo_discount.append(df_seller_post.iloc[idx]["sum(TOTAL_PROMO_DISCOUNT)"])
            post_dec_base_charge.append(df_seller_post.iloc[idx]["sum(TOTAL_BASE_CHARGE)"])

        else:
            post_dec_list_items.append(0)
            post_dec_promo_discount.append(0)
            post_dec_base_charge.append(0)

    df["post_dec_promo_discount"] = post_dec_promo_discount
    df["post_dec_base_charge"] = post_dec_base_charge
    df["post_dec_list_items"] = post_dec_list_items

    df["success"] = [1 if df.iloc[idx]["dec_list_items"] >= 1.5 * df.iloc[idx]["pre_dec_list_items"] and df.iloc[idx][
        "post_dec_list_items"] else 0 for idx in range((df.shape[0]))]

    df['SELLER_DATE_OF_REGISTRATION'] = pd.to_datetime(df['SELLER_DATE_OF_REGISTRATION'])
    df['SELLER_SPS_PROGRAM'] = df['SELLER_SPS_PROGRAM'].astype(str)
    df['SELLER_ZONE'] = df['SELLER_ZONE'].astype(str)

    return df
