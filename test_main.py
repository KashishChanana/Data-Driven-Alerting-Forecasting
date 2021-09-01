import argparse
import sys

from connection import *
from query.run_query import *
from utils.preprocess import *
from query.generate_query import *
from utils.evaluate import *
from run import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Connecting to Database via Proxy')
parser.add_argument('--connect', help='Do you want to set up a new connection [y/n] ?')
parser.add_argument('--feecode', help='Enter Fee Code to be Analyzed', type=str)
parser.add_argument('--siteid', help='Enter Listing Site ID', type=str)
parser.add_argument('--multi', help='Do you want to perform multivariate forecasting [y/n] ?')
parser.add_argument('--hourly', help='The number of hours you want to aggregate the data on', type=str)
parser.add_argument('--WoW', help='Week on Week Analysis [y/n]', type=str)
parser.add_argument('--substier', help='Analysis by SUBS_TIER [y/n]', type=str)
args = parser.parse_args()

# Set up a connection
if args.connect == 'y':
    connect()

# Initialize variables
fee_code = args.feecode
listing_site_id = args.siteid
multi = True if args.multi == 'y' else False
hours = args.hourly


# Query the data, create and clean dataframes
if args.substier:

    query, fee_column = generate_query_multi_substier(listing_site_id, fee_code)
    df = run_query(query,
                   columns=['datetime', fee_column, 'total_sale_price', 'total_promo_discount', 'subs_tier'])
    df = clean(df, columns=[fee_column, "total_sale_price", 'total_promo_discount', 'subs_tier'])

    if float(args.substier) in df['subs_tier'].unique():
        df_uni = df[df['subs_tier'] == float(args.substier)]
        plot(df_uni['datetime'], df_uni[fee_column],
             "FEE_CODE = " + fee_code + " , LISTING_SITE_ID = " + listing_site_id + " and SUBS_TIER = " + str(
                 args.substier) + " Trends by Date", "Date",
             fee_column)

        df_train, df_test = train_test_split(df_uni)

        run_model("Regression", df, df_train, df_test, fee_column, fee_code, listing_site_id)

    else:
        for uni in df['subs_tier'].unique():
            df_uni = df[df['subs_tier'] == uni]
            # print(df_uni.head())
            plot(df_uni['datetime'], df_uni[fee_column],
                 "FEE_CODE = " + fee_code + " , LISTING_SITE_ID = " + listing_site_id + " and SUBS_TIER = " + str(uni) + " Trends by Date", "Date",
                 fee_column)

            df_train, df_test = train_test_split(df_uni)

            run_model("Regression", df, df_train, df_test, fee_column, fee_code, listing_site_id)

    sys.exit()

elif args.hourly or args.WoW:
    query, fee_column = generate_query_multi_hour(listing_site_id, fee_code)
    df = run_query(query, columns=['datetime', 'hour', fee_column, 'total_sale_price', 'total_promo_discount'])
    df = clean(df, columns=[fee_column, 'hour', "total_sale_price", 'total_promo_discount'])
    if args.WoW == 'y':
        df = aggregate(df, WoW=True)
    else:
        df = aggregate(df, args.hourly)

elif args.multi == 'y':

    query, fee_column = generate_query_multi(listing_site_id, fee_code)
    df = run_query(query, columns=['datetime', fee_column, 'total_sale_price', 'total_promo_discount'])
    df = clean(df, columns=[fee_column, "total_sale_price", 'total_promo_discount'])

else:

    query, fee_column = generate_query(listing_site_id, fee_code)
    df = run_query(query, columns=['datetime', fee_column])
    df = clean(df, columns=[fee_column])

# Split the data into train and test dataframes
df_train, df_test = train_test_split(df)

# Plot observed trends
plot(df['datetime'], df[fee_column], "FEE_CODE = " + fee_code + " and LISTING_SITE_ID = " + listing_site_id + " Trends by Date", "Date", fee_column)

# Input model name
model_name = input("Enter the Model Name: ")

# Get predictions from the above specified model
predictionsDict = run_model(model_name, df, df_train, df_test, fee_column, fee_code, listing_site_id)

# Evaluate model
evaluationsDict = rmse(predictionsDict, df_test, fee_column)

# Compare and contrast between models
bar_plot(evaluationsDict)
