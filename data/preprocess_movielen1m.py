import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from gen_sequence import split_train_val_test_by_time


def gen_movielen_1m_data():
    path = "movielens1m/"
    if not os.path.exists(path):
        os.mkdir(path)
    root_path = './raw_data/'
    users = pd.read_csv(
        root_path + "ml_1m/users.dat",
        sep="::",
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    )

    ratings = pd.read_csv(
        root_path + "ml_1m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(
        root_path + "ml_1m/movies.dat", sep="::", names=["movie_id", "title", "genres"], encoding="ISO-8859-1"
    )


    ## Movies
    movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
    movies.year = pd.Categorical(movies.year)
    movies["year"] = movies.year.cat.codes
    ## Users
    users.sex = pd.Categorical(users.sex)
    users["sex"] = users.sex.cat.codes

    users.age_group = pd.Categorical(users.age_group)
    users["age_group"] = users.age_group.cat.codes

    users.occupation = pd.Categorical(users.occupation)
    users["occupation"] = users.occupation.cat.codes

    users.zip_code = pd.Categorical(users.zip_code)
    users["zip_code"] = users.zip_code.cat.codes

    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film_Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci_Fi",
        "Thriller",
        "War",
        "Western",
    ]
    for genre in genres:
        movies[genre] = movies["genres"].apply(
            lambda values: int(genre in values.split("|"))
        )
    data = pd.merge(ratings, movies, on='movie_id')
    data = data[data.rating != 3]
    data['label'] = (data.rating > 3).astype('int')

    data.drop(['rating', 'title', 'genres'], axis=1, inplace=True)

    lbes = {feature + 's': LabelEncoder() for feature in data.columns if feature != 'user_id'}
    for feature in lbes.keys():
        data[feature[:-1]] = lbes[feature].fit_transform(data[feature[:-1]])
    lbes2 = {feature: LabelEncoder() for feature in users.columns}
    for feature in lbes2.keys():
        users[feature] = lbes2[feature].fit_transform(users[feature])
    lbes = dict(lbes, **lbes2)
    with open(path+"lbes.pkl", 'wb') as file:
        pickle.dump(lbes, file)

    mms = MinMaxScaler()
    with open(path + "mms.pkl", 'wb') as file:
        pickle.dump(mms, file)

    data_group = data.sort_values(by=["timestamp"]).groupby("user_id")

    ratings_data = pd.DataFrame(
        data={
            "user_id": list(data_group.groups.keys()),
            "labels": list(data_group.label.apply(list)),
            "movie_ids": list(data_group.movie_id.apply(list)),
            "timestamps": list(data_group.timestamp.apply(list)),
            "years": list(data_group.year.apply(list)),
            "Actions": list(data_group.Action.apply(list)),
            "Adventures": list(data_group.Adventure.apply(list)),
            "Animations": list(data_group.Animation.apply(list)),
            "Childrens": list(data_group.Children.apply(list)),
            "Comedys": list(data_group.Comedy.apply(list)),
            "Crimes": list(data_group.Crime.apply(list)),
            "Documentarys": list(data_group.Documentary.apply(list)),
            "Dramas": list(data_group.Drama.apply(list)),
            "Fantasys": list(data_group.Fantasy.apply(list)),
            "Film_Noirs": list(data_group.Film_Noir.apply(list)),
            "Horrors": list(data_group.Horror.apply(list)),
            "Musicals": list(data_group.Musical.apply(list)),
            "Mysterys": list(data_group.Mystery.apply(list)),
            "Romances": list(data_group.Romance.apply(list)),
            "Sci_Fis": list(data_group.Sci_Fi.apply(list)),
            "Thrillers": list(data_group.Thriller.apply(list)),
            "Wars": list(data_group.War.apply(list)),
            "Westerns": list(data_group.Western.apply(list)),
        }
    )

    # data = pd.merge(ratings_data, users, on='user_id')

    sequence_length = 8
    step_size = 1

    def create_sequences(values, window_size, step_size):
        sequences = []
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                seq = values[-window_size:]
                if len(seq) == window_size:
                    sequences.append(seq)
                break
            sequences.append(seq)
            start_index += step_size
        return sequences

    for col in ratings_data.columns:
        if col[-1] == 's':
            ratings_data[col] = ratings_data[col].apply(
                lambda xs: create_sequences(xs, sequence_length, step_size)
            )
    ratings_data = ratings_data.explode([col for col in ratings_data.columns if col[-1] == 's'], ignore_index=True)

    ratings_data['label'] = ratings_data['labels'].apply(lambda x: x[-1])
    ratings_data.drop('labels', axis=1, inplace=True)

    ratings_data = pd.merge(ratings_data, users, on='user_id')

    ratings_data['cur_time'] = ratings_data['timestamps'].apply(lambda x: x[-1])
    train, val, test = split_train_val_test_by_time(ratings_data, 'user_id', 'cur_time')
    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_movielen_1m_data()