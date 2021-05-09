import numpy as np
import pandas as pd
import sys
import os
import ast
import matplotlib
import matplotlib.pyplot as plt
import re
from datetime import datetime

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

# Evaluate these columns as lists
column_conversion = {
    "genre": ast.literal_eval,
    "actors": ast.literal_eval
}

columns = column_types.keys()

script_folder = os.path.dirname(os.path.abspath(__file__))
finaldata_path = os.path.join(script_folder, '..\\Data\\FinalData.csv')

# Main dataframe of all data
main_df = pd.read_csv(finaldata_path, usecols=columns, dtype=column_types,
                      converters=column_conversion)


def recommend_movie(user_selection):
    # Make a copy of the imported data, don't overwrite so we
    # can call function many times without re-importing
    df = main_df.copy()

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
            # Handle errors with movies not in data
            print("ERROR: ", movie, " NOT FOUND")
            user_selection.remove(movie)

    print("Input Movies:")
    for movie in input_movies:
        print(movie)

    print("\n\n")

    # ----------------Calculations from inputs---------------------
    # ---Average Rating---
    # Get the average rating from each movie, store in a list
    ratings = []
    for movie in user_selection:
        ratings.append(df.loc[df['imdb_title_id'] == movie]['avg_vote'].item())

    # Calculate mean and std of average rating
    rating_mean = np.mean(ratings)
    rating_std = np.std(ratings)

    # ---Num Ratings---
    # Get the number of ratings from each movie, store in a list
    num_ratings = []
    for movie in user_selection:
        num_ratings.append(
            df.loc[df['imdb_title_id'] == movie]['votes'].item())

    num_ratings_mean = np.mean(num_ratings)
    num_ratings_std = np.std(num_ratings)

    # ---Year---
    # Get the release year from each movie, store in a list
    years = []
    for movie in user_selection:
        years.append(df.loc[df['imdb_title_id'] == movie]['year'].item())

    year_mean = np.mean(years)
    year_std = np.std(years)

    # ---Budget---
    # Get the budget of each movie, store in a list
    budgets = []
    for movie in user_selection:
        budget = df.loc[df['imdb_title_id'] == movie]['budget'].item()
        if budget > 0:
            # Don't consider if no budget given
            budgets.append(budget)

    budgets_mean = np.mean(budgets)
    budgets_std = np.std(budgets)

    # ---Genre---
    genre_count = {}

    # Each movie can have multiple genres
    # Count the number of each genre as it appears in all of the input movies
    # Store in dict
    for movie in user_selection:
        for genre in (df.loc[df['imdb_title_id'] == movie]['genre'].item()):
            if genre not in genre_count:
                genre_count[genre] = 1
            else:
                genre_count[genre] += 1

    # ---Director---
    director_count = {}

    # Each movie has one director, create a dict for the counts
    # of how many times a director appears in the input movies
    for movie in user_selection:
        director = df.loc[df['imdb_title_id'] == movie]['director'].item()

        if director not in director_count:
            director_count[director] = 1
        else:
            director_count[director] += 1

    # ---Actors---
    actor_count = {}

    # Each movie can have many actors
    # Count the number of each actor as it appears in all of the input movies
    # Only consider the first two actors, as these are the leading actors and
    # who people will see the most
    for movie in user_selection:
        # Get first two listed actors, add them to dict
        for actor in (df.loc[df['imdb_title_id'] == movie]['actors'].item())[0:2]:
            if actor not in actor_count:
                actor_count[actor] = 1
            else:
                actor_count[actor] += 1

    # ----------------Drop user choices from df-----------------------
    # Drop the input movies from the dataframe
    for movie in user_selection:
        df = df[~df.imdb_title_id.str.contains(movie, na=False)]

    df = df.reset_index(drop=True)

    # ---------------------Point Calculations------------------------
    # ---Average Rating---
    # Drop movies with avg_ratings not within 2 sigma of mean
    df = df[df['avg_vote'] <= rating_mean + (2*rating_std)]
    df = df[df['avg_vote'] >= rating_mean - (2*rating_std)]
    df = df.reset_index(drop=True)

    # We now have a range given by: (mean-2*std) to (mean+2*std)
    # Assign point range where moves at lowest endpoint get 0 pts
    # And movies at high endpoint get 2 pts
    old_high = rating_mean + 2*rating_std
    old_low = rating_mean - 2*rating_std
    old_range = (old_high - old_low)

    new_high = 2.0
    new_low = 0.0
    new_range = (new_high - new_low)

    for i in df.index:
        avg_vote = df['avg_vote'][i]
        scaled_points = (((avg_vote - old_low) * new_range) /
                         old_range) + new_low
        df['points'][i] += scaled_points

    # ---Number of Ratings---
    # Drop movies with num_ratings not within 2 sigma
    df = df[df['votes'] <= num_ratings_mean + (2*num_ratings_std)]
    df = df[df['votes'] >= num_ratings_mean - (2*num_ratings_std)]
    df = df.reset_index(drop=True)

    # If num_ratings > half of lowest num_ratings, and < twice highest num_ratings
    # give it an extra point
    # Do 2nd lowest and 2nd highest instead
    num_ratings.sort()
    second_lowest = num_ratings[1]
    second_highest = num_ratings[len(num_ratings)-2]
    for i in df.index:
        if df['votes'][i] >= .5*second_lowest and df['votes'][i] <= 2*second_highest:
            df['points'][i] += 1

    # We have a range given by: (mean-2*std) to (mean+2*std)
    # Assign point range where moves at lowest endpoint get 0 pts
    # And movies at high endpoint get 2 pts
    old_high = 2*max(num_ratings)
    old_low = .5*min(num_ratings)
    old_range = (old_high - old_low)

    new_high = 1.0
    new_low = 0.0
    new_range = (new_high - new_low)

    for i in df.index:
        votes = df['votes'][i]
        scaled_points = (((votes - old_low) * new_range) /
                         old_range) + new_low
        df['points'][i] += scaled_points

    # ---Year---
    # Drop movies with year not within 2 sigma
    df = df[df['year'] <= year_mean + (2*year_std)]
    df = df[df['year'] >= year_mean - (2*year_std)]
    df = df.reset_index(drop=True)

    # ---Budget---
    # Drop movies with budgets not within 2 sigma
    df = df[df['budget'] <= budgets_mean + (2*budgets_std)]
    df = df[df['budget'] >= budgets_mean - (2*budgets_std)]
    df = df.reset_index(drop=True)

    # If budgets > half of lowest budgets, and < twice highest budgets
    # give it an extra point
    budgets.sort()
    second_lowest = budgets[1]
    second_highest = budgets[len(budgets)-2]
    for i in df.index:
        if df['budget'][i] >= .5*second_lowest and df['budget'][i] <= 2*second_highest:
            df['points'][i] += 1

    # ---Genres---
    # Assign points based on genre_count (if sci-fi = 3, and the movie is sci
    # fi, then given them 3 points)
    genres = genre_count.keys()
    for i in df.index:
        for genre in df["genre"][i]:
            if genre in genres:
                df["points"][i] += genre_count[genre]

    df = df.reset_index(drop=True)

    # ---Director---
    # Give movies with directors who appear in the count dict points
    # Points based on how many times they appeared in input movies
    directors = director_count.keys()
    for i in df.index:
        for director in df["director"][i]:
            if director in directors:
                df["points"][i] += (director_count[director])

    # ---Actor---
    # Give movies with actors who appear in the count dict points
    # Points based on how many times they appeared in input movies
    actors = actor_count.keys()
    for i in df.index:
        for actor in df["actors"][i][0:2]:
            if actor in actors:
                df["points"][i] += (actor_count[actor]) - 1

    # -------------Get top 5 movies to recommend-------------------
    df = df.sort_values('points', ascending=False)
    df = df.reset_index(drop=True)

    recommended_movies = df[['original_title', 'points']].head(5)
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
    f.write("Ratings: " + str(ratings) + "\n")
    f.write("Rating Avg: " + str(rating_mean) + "\n")
    f.write("Rating Std: " + str(rating_std) + "\n")

    f.write("Num Ratings: " + str(num_ratings) + "\n")
    f.write("Num Ratings Avg: " + str(num_ratings_mean) + "\n")
    f.write("Num Ratings Std: " + str(num_ratings_std) + "\n")

    f.write("Years: " + str(years) + "\n")
    f.write("Years Avg: " + str(year_mean) + "\n")
    f.write("Years Std: " + str(year_std) + "\n")

    f.write("Budgets: " + str(budgets) + "\n")
    f.write("Budgets Avg: " + str(budgets_mean) + "\n")
    f.write("Budgets Std: " + str(budgets_std) + "\n")

    f.write("Genres: " + str(genre_count) + "\n")

    f.write("Directors: " + str(director_count) + "\n")

    f.write("Actors: " + str(actor_count) + "\n")

    f.write("\nRecommended movies:\n")
    for movie in output_movies:
        # Movie Title
        f.write(movie[0] + '\n')
        # Points
        # f.write(str(round(movie[1], 2)) + "\n")

    f.write("\n-----------------------------------\n\n")

    f.close()

    # ----------------Output Graphs--------------------
    # Print Histogram of points
    df.hist(column='points')
    plt.title('Point Distribution')
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


# Get five IMDb links from user
print("Please input IMDb links for your five favorite movies")
movie_num = 1
movie_ids = []
while movie_num <= 5:
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

recommend_movie(movie_ids)

# Test set of my favorite movies
# https://www.imdb.com/title/tt2713180/
# https://www.imdb.com/title/tt1375666/
# https://www.imdb.com/title/tt0816692/
# https://www.imdb.com/title/tt0206634/
# https://www.imdb.com/title/tt0137523/
