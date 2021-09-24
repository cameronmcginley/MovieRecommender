import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import re
from datetime import datetime

def get_movie_dataset():
    # Import dataframe from the FinalData.csv
    column_types = {
        "imdb_title_id": str,
        "original_title": str,
        "year": float,
        "genre": object,
        "director": str,
        "actors": object,
        "avg_vote": float,
        "votes": int,
        "budget": float
    }

    # Converters
    column_conversion = {
        "genre": ast.literal_eval,
        "actors": ast.literal_eval
    }

    script_folder = os.path.dirname(os.path.abspath(__file__))
    final_data_path = os.path.join(script_folder, '..\\Data\\FinalData.csv')

    # Main dataframe of all data
    columns = column_types.keys()
    return pd.read_csv(final_data_path, usecols=columns, dtype=column_types,
                        converters=column_conversion)


def get_movie_inputs(n):
    # Get n IMDb links from user
    print("Please input IMDb links for your five favorite movies")
    movie_num = 1
    movie_ids = []
    while movie_num <= n:
        link = input("Input IMDb link for movie #" + str(movie_num) + ": ")

        # Movie ID comes between "title/" and "/" in the IMDb link
        split_link = re.search('title/(.+?)/', link)
        if split_link:
            movie_id = split_link.group(1)

        # Add the id to list if it exists, otherwise print error and repeat prompt
        if split_link:
            # Append to movie_ids list if not a duplicate
            if movie_id not in movie_ids:
                movie_ids.append(movie_id)
                movie_num += 1
            else:
                print("Do not enter duplicate movies")
        else:
            print("Link must contain title ID of format \"title/tt#######/\"")


    print()
    # Get amount of movies to recommend
    num = 0
    while num <= 0 or num > 100:
        num = input("How many movies to recommend: ")

        try:
            num = int(num)
        except:
            num = 0
            print("Try again")
            continue


    print()
    # Get importances from user
    # Defaults
    importances = {
        "rating": 2,
        "num_ratings": 2,
        "year": 2,
        "budgets": 2,
        "genre_count": 2,
        "director_count": 2,
        "actor_count": 2
    }

    do_customize = input("Modify weights (y/n): ")
    while do_customize.lower() != 'n' and do_customize.lower() != 'y':
        do_customize = input("Modify weights (y/n): ")

    if do_customize.lower() == 'y':
        print("1 = very important, 2 = moderate, 3 = not important")
        for key in importances:
            importances[key] = input(key + " (1-3): ")
            while importances[key] != '1' and importances[key] != '2' and importances[key] != '3':
                print("Try again")
                importances[key] = input(key + " (1-3): ")

            importances[key] = int(importances[key])

    return [movie_ids, num, importances]


# Gets list of values of column_name from each movie in movie_ids
def get_stats_list(df, movie_ids, column_name):
    return_list = []
    for movie in movie_ids:
        value = df.loc[df['imdb_title_id'] == movie][column_name].item()
        if value:
            return_list.append(value)

    return return_list


def add_counts(val, dict):
    if val not in dict:
        dict[val] = 1
    else:
        dict[val] += 1

    return dict


# Gets various statistics from the user input movies
# Returns a dictionary with all of the data needed by the recommender
def get_input_stats(movie_ids, df):
    # Lists of values of the input movies given by column name
    ratings = get_stats_list(df, movie_ids, "avg_vote")
    num_ratings = get_stats_list(df, movie_ids, "votes")
    years = get_stats_list(df, movie_ids, "year")
    budgets = get_stats_list(df, movie_ids, "budget")

    genre_count = {}
    director_count = {}
    actor_count = {}

    # Get counts of genres, directors, and actors for all input movies
    for movie in movie_ids:
        for genre in (df.loc[df['imdb_title_id'] == movie]["genre"].item()):
            genre_count = add_counts(genre, genre_count)

        director = df.loc[df['imdb_title_id'] == movie]["director"].item()
        director_count = add_counts(director, director_count)

        for actor in (df.loc[df['imdb_title_id'] == movie]["actors"].item())[0:2]:
            actor_count = add_counts(actor, actor_count)

    return {
        "ratings" : ratings,
        "rating_mean": np.mean(ratings),
        "rating_std": np.std(ratings),
        "num_ratings": num_ratings,
        "num_ratings_mean": np.mean(num_ratings),
        "num_ratings_std": np.std(num_ratings),
        "years": years,
        "year_mean": np.mean(years),
        "year_std": np.std(years),
        "budgets": budgets,
        "budgets_mean": np.mean(budgets),
        "budgets_std": np.std(budgets),
        "genre_count": genre_count,
        "director_count": director_count,
        "actor_count": actor_count
    }


# Drops data outside of a certain range of mean
def drop_data_range(df, df_name, stats_name, stats, importance):
    # Important for stats data to match this format
    type_mean = stats_name + "_mean"
    type_std = stats_name + "_std"

    # default = 2
    std_range = importance

    # drop data outside of this range
    df = df[df[df_name] <= stats[type_mean] + (std_range * stats[type_std])]
    df = df[df[df_name] >= stats[type_mean] - (std_range * stats[type_std])]
    df = df.reset_index(drop=True)
    return df


def calculate_points(df, df_name, stats_name, stats, importances, max_pts, award_type):
    # Add bonus to max pts
    max_pts = max_pts + importances[stats_name]
    if max_pts < 0 or importances[stats_name] == -1: max_pts = 0
    min_pts = 0

    if award_type == "scale":
        pts_range = (max_pts - min_pts)

        old_high = df[df_name].max()
        old_low = df[df_name].min()
        old_range = (old_high - old_low)

        # Award movies with higher value a higher number of points
        # Linear scale between min and max based on their old value
        for i in df.index:
            value = df[df_name][i]
            scaled_points = (((value - old_low) * pts_range) /
                            old_range) + min_pts
            df['points'][i] += scaled_points

    elif award_type == "normal":
        # Not exactly normal, but values within a distance of min and max values
        # are awarded more
        stats[stats_name].sort()
        second_lowest = stats[stats_name][1]
        second_highest = stats[stats_name][len(stats[stats_name])-2]
        for i in df.index:
            # Within half of 2nd least and double of 2nd most
            if df[df_name][i] >= .5*second_lowest and df[df_name][i] <= 2*second_highest:
                df['points'][i] += max_pts

    elif award_type == "counts":
        # Assign points based on counts (if sci-fi = 3, and the movie is sci
        # fi, then give them 3 points plus 3*max_pts)
        genres = stats[stats_name].keys()
        for i in df.index:
            for genre in df[df_name][i]:
                if genre in genres:
                    points = stats[stats_name][genre] * max_pts #+ stats[stats_name][genre]*max_pts
                    df["points"][i] += points

    return df


def recommend_movie(n, user_selection, df, stats, importances):
    # Start tracking points for movies inside a new column
    df['points'] = 0
    df['points'] = df['points'].astype('float')

    # Print the names of movies given by a user
    input_movies = []
    for movie in user_selection:
        try:
            input_movies.append(
                df.loc[df['imdb_title_id'] == movie]['original_title'].item())
        except:
            # If movie not in dataset
            print("ERROR: ", movie, " NOT FOUND")
            user_selection.remove(movie)

    print("Input Movies:")
    for movie in input_movies:
        print(movie)

    # Drop the input movies from the dataframe
    for movie in user_selection:
        df = df[~df.imdb_title_id.str.contains(movie, na=False)]

    df = df.reset_index(drop=True)

    # -----Drop data outside of range specified by importance (std from mean)-----
    df = drop_data_range(df, "avg_vote", "rating", stats, importances["rating"])
    df = drop_data_range(df, "votes", "num_ratings", stats, importances["num_ratings"])
    df = drop_data_range(df, "year", "year", stats, importances["year"])
    df = drop_data_range(df, "budget", "budgets", stats, importances["budgets"])

    # Convert importances into bonus points
    # importance of 1 = 1 bonus point, of 2 = 0, of 3 = -1
    for key in importances:
        if importances[key] == 2:
            importances[key] = 0
        elif importances[key] == 3:
            importances[key] = -1


    # ---------------------Point Calculations------------------------
    # Rating uses scale to skew points to more favored movies
    df = calculate_points(df, "avg_vote", "rating", stats, importances, 1, award_type="scale")

    # Num of ratings and budgets awards points to more similar values
    df = calculate_points(df, "votes", "num_ratings", stats, importances, 1, award_type="normal")
    df = calculate_points(df, "budget", "budgets", stats, importances, .5, award_type="normal")

    # The following awards points based on shared occurences (same genres, etc.)
    df = calculate_points(df, "genre", "genre_count", stats, importances, .4, award_type="counts")
    df = calculate_points(df, "director", "director_count", stats, importances, .3, award_type="counts")
    df = calculate_points(df, "actors", "actor_count", stats, importances, .2, award_type="counts")


    # -------------Get top n movies to recommend-------------------
    df = df.sort_values('points', ascending=False)
    df = df.reset_index(drop=True)

    recommended_movies = df[['original_title', 'points']].head(n)
    print(recommended_movies)
    output_movies = []
    for i in recommended_movies.index:
        output_movies.append(
            (recommended_movies.loc[i, 'original_title'],
             recommended_movies.loc[i, 'points']))


    # --------------------Print to log file-----------------------
    f = open("results_log.txt", "a")

    f.write("User input:\n")
    for movie in input_movies:
        f.write(movie + "\n")

    f.write("\nStats:\n")
    f.write("Ratings: " + str(stats["ratings"]) + "\n")
    f.write("Rating Avg: " + str(stats["rating_mean"]) + "\n")
    f.write("Rating Std: " + str(stats["rating_std"]) + "\n")

    f.write("Num Ratings: " + str(stats["num_ratings"]) + "\n")
    f.write("Num Ratings Avg: " + str(stats["num_ratings_mean"]) + "\n")
    f.write("Num Ratings Std: " + str(stats["num_ratings_std"]) + "\n")

    f.write("Years: " + str(stats["years"]) + "\n")
    f.write("Years Avg: " + str(stats["year_mean"]) + "\n")
    f.write("Years Std: " + str(stats["year_std"]) + "\n")

    f.write("Budgets: " + str(stats["budgets"]) + "\n")
    f.write("Budgets Avg: " + str(stats["budgets_mean"]) + "\n")
    f.write("Budgets Std: " + str(stats["budgets_std"]) + "\n")

    f.write("Genres: " + str(stats["genre_count"]) + "\n")

    f.write("Directors: " + str(stats["director_count"]) + "\n")

    f.write("Actors: " + str(stats["actor_count"]) + "\n")

    f.write("\nRecommended movies:\n")
    for movie in output_movies:
        # Movie Title
        f.write(movie[0] + '\n')
        # Points
        f.write(str(round(movie[1], 2)) + "\n")

    f.write("\n-----------------------------------\n\n")

    f.close()

    # ----------------Output Graphs--------------------
    # Print Histogram of points
    df.hist(column='points')
    plt.title('Points of All Non-Dropped Movies')
    plt.xlabel('Points')
    plt.ylabel('Number of Movies')
    plt.show()

    # Define y range for bar chart
    y_diff = df['points'][0] - df['points'][9]
    y_max = df['points'][0]
    y_min = df['points'][9] - (y_diff/2)

    # Print bar chart of points from top 10 movies
    (df[['points']].head(10)).plot.bar()
    axes = plt.gca()
    axes.set_ylim([y_min, y_max])
    plt.title('Points of Top Ten Movies')
    plt.xlabel('Movie Rank')
    plt.ylabel('Points')
    plt.show()


# Test set of my favorite movies
"""
https://www.imdb.com/title/tt2713180/
https://www.imdb.com/title/tt1375666/
https://www.imdb.com/title/tt0816692/
https://www.imdb.com/title/tt0206634/
https://www.imdb.com/title/tt0137523/
"""

inputs = get_movie_inputs(5)
movie_ids = inputs[0]
num_to_recommend = inputs[1]
importances = inputs[2]

# Gets dataset from FinalData.csv
# Must run DataCleaner.py first
all_movie_data = get_movie_dataset()

input_stats = get_input_stats(movie_ids, all_movie_data)

recommend_movie(num_to_recommend, movie_ids, all_movie_data, input_stats, importances)
