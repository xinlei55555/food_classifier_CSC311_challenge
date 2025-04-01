import csv
from thefuzz import fuzz

"""
    Returns the list of movies grouped together from the csv's 6th column.
    Two movies are grouped together if fuzz.ratio() > 65.
"""
def extract_movies(csv_file):
    movies_list = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if it exists
        for index, row in enumerate(reader, start=1):
            if len(row) >= 6:  # Ensure the row has at least 6 columns
                movie_name = row[5].lower()
                movies_list.append(movie_name)

    movie_directory = {}
    for i in range(len(movies_list)):
        movie = movies_list[i]

        best_similarity = -1
        movie_d_best = None
        for movie_d in movie_directory:
            r = fuzz.ratio(movie, movie_d)
            if r > best_similarity:
                best_similarity = r
                movie_d_best = movie_d

        if best_similarity > 65:
            movie_directory[movie_d_best].append(movie)
        else:
            movie_directory[movie] = [movie]

    most_popular = []

    for k, v in zip(movie_directory.keys(), movie_directory.values()):
        if len(k) < 20:
            most_popular.append((k, len(v)))

    return most_popular

if __name__ == '__main__':
    movies_list = extract_movies('../data/cleaned_data_combined_modified.csv')
    movies_list.sort(key=lambda x: x[1])
    movies_list.reverse()

    print(movies_list[:20])
    #for movie in movies_list:
    #    if len(movie) < 20:
    #        print(movie)