def generate_query(listing_site_id, fee_code):
    """
    Generates query.

    Parameters
    ----------------
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed

    Returns
    ----------------
    Query and fee column name.
    """

    fee_column = 'total_fee_' + fee_code
    query = "select toDate(CHARGE_DATE) AS date ,sum(AMOUNT_CHARGED) as " + fee_column + " from @Database where "\
                                                                                         "LISTING_SITE_ID =" + \
            listing_site_id + " and FEE_CODE = " + fee_code + "  group " \
                                                              "by date order by date ASC "
    return query, fee_column

def generate_query_multi(listing_site_id, fee_code):

    """
    Generates query with multivariate features.

    Parameters
    ----------------
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed

    Returns
    ----------------
    Query and fee column name.
    """

    # TOTAL_BASE_FEE_CHARGE, TOTAL_PROMO_DISCOUNT

    fee_column = 'total_fee_' + fee_code
    query = "select toDate(CHARGE_DATE) AS date ,sum(AMOUNT_CHARGED) as " + fee_column + " , sum(SALE_PRICE) as total_sale_price," \
            + "sum(TOTAL_PROMO_DISCOUNT) as total_promo_discount from @Database where LISTING_SITE_ID =" + \
            listing_site_id + " and FEE_CODE = " + fee_code + "  group " \
                                                              "by date order by date ASC "

    return query, fee_column

def generate_query_multi_hour(listing_site_id, fee_code):

    """
    Generates query by the hour.

    Parameters
    ----------------
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed

    Returns
    ----------------
    Query and fee column name.
    """

    fee_column = 'total_fee_' + fee_code
    query = "select toDate(CHARGE_DATE) as date, toHour(CHARGE_DATE) as hour ,sum(AMOUNT_CHARGED) as " + fee_column + " , sum(SALE_PRICE) as total_sale_price," \
            + "sum(TOTAL_PROMO_DISCOUNT) as total_promo_discount from @Database where LISTING_SITE_ID =" + \
            listing_site_id + " and FEE_CODE = " + fee_code + "  group " \
                                                              "by date, hour order by date, hour ASC "

    return query, fee_column

def generate_query_multi_substier(listing_site_id, fee_code):
    """
    Generates query aggregated by substier.

    Parameters
    ----------------
    :param listing_site_id: listing site id on eBay
    :param fee_code: code of fee_type being analyzed

    Returns
    ----------------
    Query and fee column name.
    """

    fee_column = 'total_fee_' + fee_code
    query = "select toDate(CHARGE_DATE) as date ,sum(AMOUNT_CHARGED) as " + fee_column + " , sum(SALE_PRICE) as total_sale_price," \
            + "sum(TOTAL_PROMO_DISCOUNT) as total_promo_discount, SUBS_TIER as subs_tier from @Database where LISTING_SITE_ID =" + \
            listing_site_id + " and FEE_CODE = " + fee_code + "  group " \
                                                              "by date, subs_tier order by date, subs_tier ASC "

    return query, fee_column



