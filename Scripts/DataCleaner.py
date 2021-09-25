import pandas as pd
import os
import re
from datetime import datetime
from currency_converter import CurrencyConverter
from numpy import nan

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)


# Prints time log to console at start of operation
def log_s(msg):
    print("[", datetime.now(), "] ", msg, sep="")


# Prints time log to console after operation finished
def log_f():
    print("[", datetime.now(), "] ", "Finished\n", sep="")


def main():
    # Parameters for dropping/keeping data
    min_num_reviews = 1000
    english_only = True


    # -------------------Read in data to dataframe-------------------
    log_s("Reading \"IMDb movies.csv\" to DF...")

    # Project folders and data file
    script_folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_folder, '..\\Data')
    imdb_movies_path = os.path.join(data_path, 'IMDb movies.csv')

    # Must manually specify types, takes lots of memory for pandas to guess
    # Only choosing some columns from the input data file
    column_types = {
        "imdb_title_id": str,
        "original_title": str,
        "year": str,
        "genre": object,
        "language": str,
        "director": str,
        "actors": object,
        "avg_vote": float,
        "votes": int,
        "budget": str,
    }

    column_conversion = {
        # Replace string with list of strings for these columns
        "genre": lambda x: x.split(', '),
        "actors": lambda x: x.split(', '),
        "language": lambda x: x.split(', ')
    }

    # Select which columns to use from the data file
    columns = column_types.keys()

    df = pd.read_csv(imdb_movies_path, usecols=columns, dtype=column_types,
                    converters=column_conversion)

    log_f()


    # ------------Drop rows if <1000 reviews-----------------------
    # Decided to exclude barely known movies, as these will almost never
    # be recommended to a user anyways
    log_s("Dropping movies with < " + str(min_num_reviews) + " reviews...")

    # Keep only rows with >= 1000 votes
    df = df[df['votes'] >= min_num_reviews]
    df = df.reset_index(drop=True)

    log_f()

    # ------------Drop rows if not English-------------------------
    if english_only:
        log_s("Dropping non-english movies...")

        # Check if main laingauge (first index) is English, drop if not
        df = df[df['language'].str[0] == "English"]
        df = df.reset_index(drop=True)

        # Drop English column
        df.drop(columns=['language'])

        log_f()


    # -------------Convert year from str to int------------------
    # Some years have additional characters we need to remove
    log_s("Removing non-numeric chars from year, changing to int...")

    for i in df.index:
        # check if it's not already int first, saves time
        if not df["year"][i].isdigit():
            df["year"][i] = re.sub("[^0-9]", "", df["year"][i])

    # Change column type
    df["year"] = df["year"].astype("int")

    log_f()


    # -------------Convert budget from str to int------------------
    # Note: Add currency conversions
    log_s("Converting currencies to USD...")

    # Use currency converter library
    c = CurrencyConverter()

    df["budget"] = df["budget"].str.replace(" ", "")

    for i in df.index:
        # Get curr, e.g. USD, and the amount
        curr_code = re.sub('[0-9]', '', str(df["budget"][i]))
        amt = re.sub('[^0-9]', '', str(df["budget"][i]))

        # Only convert if exists and not USD already
        if curr_code and amt and curr_code != '$':
            try:
                # Everything already in correct currency code
                amt = c.convert(amt, curr_code, 'USD')
                amt = str(round(amt, 2))
            except:
                # Currency isn't supported
                amt = nan

        if amt:
            df["budget"][i] = float(amt)

    log_f()


    # -----------------Print final/cleaned data to CSV-----------------------
    log_s("Printing to Data/FinalData.csv...")

    final_data_path = os.path.join(data_path, 'FinalData.csv')
    df.to_csv(final_data_path)

    log_f()


if __name__ == '__main__':
    main()