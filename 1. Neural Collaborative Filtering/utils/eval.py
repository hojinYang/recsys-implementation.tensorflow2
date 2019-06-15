import math
import pandas as pd

class Evaluation():
    def __init__(self, subjects):
        """
        args:
          subjects:list, [test_users, test_items, test_scores, negative_users, negative_items, negative_scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        
        #the golden set
        test = pd.DataFrame({'user':test_users,
                            'item':test_items,
                            'score':test_scores})


        neg = pd.DataFrame({'user':neg_users,
                           'item':neg_items,
                           'score': neg_scores})
        test['is_answer'] = True
        neg['is_answer'] = False

        self._num_answers = test[['user','item']].groupby('user').count().reset_index().rename(columns={"item":'num_answers'})

        full = test.append(neg).reset_index(drop=True)
        full['rank'] = full.groupby('user')['score'].rank(method = 'first', ascending = False)
        full.sort_values(['user', 'rank'], inplace=True)

        self._full = full
    
    def _get_correct_counter_k(self, k):
        top_k = self._full[self._full['rank']<=k]
        ans_in_top_k = top_k[top_k['is_answer'] == True]
        correct_counter = ans_in_top_k[['user','item']].groupby('user').count().reset_index().rename(columns={"item":'num_correct'})
        correct_counter = pd.merge(self._num_answers, correct_counter,how='left')
        return correct_counter
  
    def _get_recall_k(self, correct_counter, k):
        temp = correct_counter['num_correct'] / k
        return temp.mean()

    def _get_prec_k(self, correct_counter, k):
        temp = correct_counter['num_correct'] / correct_counter['num_answers']     
        return temp.mean()

    def print_eval_score_k(self, k):
        correct_counter = self._get_correct_counter_k(k)
        recall = self._get_recall_k(correct_counter, k)
        prec = self._get_prec_k(correct_counter, k)
        print("recall@{2}:{0:.4f}, prec@{2}:{1:.4f}".format(recall,prec,k))