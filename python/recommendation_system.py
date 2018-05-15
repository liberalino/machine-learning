'''
    This code implements a recommender movie system using LightFM, a library
    that implements popular recommendation algorithms

Dependencies:
    LightFM (https://github.com/lyst/lightfm)
    numpy (pip install numpy)
    scipy (pip install scipy)

References:
    https://github.com/llSourcell/recommender_system_challenge


'''


import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens


data = fetch_movielens(min_rating = 4.0)
# Training and testing datasets
print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss='warp')
model.fit(data['train'], epochs = 30, num_threads = 2)

def sample_recommendation(model, data, user_ids):

    # Number of users and movies in training data
    n_users, n_items = data['train'].shape

    # Generate recommendations for each user
    for user_id in user_ids:

        # Movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Movies the model predicts they will like
        # The model will give a score for each movie for the given user
        scores = model.predict(user_id, np.arange(n_items))

        # Rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # Print out the results
        print('User {}'.format(user_id))
        print('\tKnow positives:')

        for x in known_positives[:3]:
            print('\t\t{}'.format(x))

        print('\tRecommended:')

        for x in top_items[:3]:
            print('\t\t{}'.format(x))

sample_recommendation(model, data, [3, 25, 450])
