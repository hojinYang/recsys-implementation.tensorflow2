import random
import pandas as pd
import numpy as np
from copy import deepcopy
import tensorflow as tf 

random.seed(0)


class SampleGenerator():
    """Construct dataset"""
  
    def __init__(self, ratings, implicit = True):
        """
        args: 
          ratings: pd.DataFrame which contains 4 columns=['userId','itemId','rating','timestamp']
          implicit: convert to implicit rating(0,1) if specified
        """
        assert'userId' in ratings.columns
        assert'itemId' in ratings.columns
        assert'rating' in ratings.columns

        self.ratings = ratings
        if implicit:
            self.preprocess_ratings = self._binarize(ratings)
        else:
            self.preprocess_ratings = self._normalize(ratings)
    
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
    
        self.num_users = len(self.user_pool)
        self.num_items = len(self.item_pool)
    
        # self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
        self.train_ratings, self.test_ratings = self._split_by_ratio(self.preprocess_ratings, 0.3)

        self.train_interacted_status = self._get_user_interact_status(self.train_ratings)
        self.test_interacted_status = self._get_user_interact_status(self.test_ratings)

        self.test_interacted_status = self._delete_interacted_items(self.test_interacted_status, self.train_interacted_status)
    
  
    def _normalize(self, ratings):
        """
        change ratings to have 0~1
        """
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating*1.0/max_rating
        return ratings
  
    def _binarize(self, ratings):
        """
        change ratings to have 0 or 1
        """
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] >0] = 1.0
        return ratings
  
    def _get_user_interact_status(self, ratings):
        '''
        return : pd.DataFrame which contains 3 columns=[userId, interacted_items, negative_items]
        '''
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        return interact_status
  
    def _split_loo(self, ratings):
        """
        leave one out: set test item as the last one
        """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _split_by_ratio(self, ratings, test_ratio):
        """
        split by ratio
        """
        train, test = np.split(ratings.sample(frac=1), [int((1-test_ratio) * len(ratings))])
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _delete_interacted_items(self, test_interacted_status, train_interacted_status):
        df = pd.merge(test_interacted_status, train_interacted_status, how='left', on='userId')
        df['negative_items'] = df['negative_items_x'] - df['interacted_items_y']
        df = df.rename(columns={'interacted_items_x':'interacted_items'})

        return df[['userId','interacted_items', 'negative_items']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.train_interacted_status[['userId','negative_items']], on='userId')
        train_ratings['negative_samples'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negative_samples[i]))
                ratings.append(float(0))      
        
        dataset = tf.data.Dataset.from_tensor_slices((users, items, ratings))
        dataset = dataset.shuffle(buffer_size=len(users)).batch(batch_size)
        return dataset

    
    def instance_test_loader(self, num_negatives, batch_size):
        test_users, test_items, negative_users, negative_items = [], [], [], []
        if num_negatives == -1:
            for row in self.test_interacted_status.itertuples():
                ni = list(row.negative_items)
                negative_users = negative_users + [int(row.userId)]*len(ni)
                negative_items = negative_items + ni
        else:
            self.test_interacted_status['negative_samples'] = self.test_interacted_status['negative_items'].apply(lambda x: random.sample(x, num_negatives))
            for row in self.test_interacted_status.itertuples():
                ni = list(row.negative_samples)
                negative_users = negative_users + [int(row.userId)]*len(ni)
                negative_items = negative_items + ni

        test_users = list(self.test_ratings['userId'])
        test_items = list(self.test_ratings['itemId'])

        test_dataset = tf.data.Dataset.from_tensor_slices((test_users, test_items)).batch(batch_size)
        negative_dataset = tf.data.Dataset.from_tensor_slices((negative_users, negative_items)).batch(batch_size)
        
        return test_dataset, negative_dataset