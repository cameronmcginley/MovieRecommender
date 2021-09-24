# Movie Recommender
**Original Data Source:** https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset 

Scripts/DataCleaner.py generates FinalData.csv

Scripts/Recommender.py handles everything from gathering user inputs, to dropping movies dissimilar to inputs, to calculating points, to outputting the final n recommendations

Point system awards points to movies based on how well they match the user's favorite/input movies in certain categories. This is done in a few different methods and distributions depending on the category. Before awarding points, data that is very unalike the inputs are dropped completely to speed things up.

There are 7 categories in total that points are based on, and users are able to input their preference of weighting for each.

Outputs printed to results_log.txt
