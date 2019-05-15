import random

import keras
import numpy as np
import pandas as pd


# TODO: consider questions without answers

class BatchGenerator(keras.utils.Sequence):
    """
    Class to ingest data from pre-processed DataFrames to model
    in form of batches of NumPy arrays
    """

    exp_mean = 30

    def __init__(self, que: pd.DataFrame, stu: pd.DataFrame, pro: pd.DataFrame,
                 batch_size: int, pos_pairs: list, nonneg_pairs: list, pro_dates: dict):
        """
        :param que: pre-processed questions data
        :param stu: pre-processed students data
        :param pro: pre-processed professionals data
        :param batch_size: actually, half of the real batch size
        Number of both positive and negative pairs present in generated batch
        :param pos_pairs: tuples of question, student and professional, which form positive pair
        (professional answered on the given question from corresponding student)
        :param nonneg_pairs: tuples of question, student and professional, which are known to form a positive pair.
        Superset of pos_pairs, used in sampling of negative pairs
        :param pro_dates: mappings from professional's id to his registration date
        """
        self.batch_size = batch_size

        # extract mappings from question's id to question's date and features
        que_ar = que.values
        self.que_feat = {que_ar[i, 0]: que_ar[i, 2:] for i in range(que_ar.shape[0])}
        self.que_time = {que_ar[i, 0]: pd.Timestamp(que_ar[i, 1]) for i in range(que_ar.shape[0])}

        self.pos_pairs = pos_pairs
        self.on_epoch_end()  # shuffle pos_pairs
        self.nonneg_pairs = {(que, stu, pro) for que, stu, pro, time in nonneg_pairs}

        # these lists are used in sampling of negative pairs
        self.ques_stus_times = [(que, stu, self.que_time[que]) for que, stu, pro, time in pos_pairs]

        self.pros = np.array([pro for que, stu, pro, time in nonneg_pairs])
        self.pros_times = np.array([pro_dates[pro] for que, stu, pro, time in nonneg_pairs])

        # simultaneously sort two arrays containing professional features
        sorted_args = np.argsort(self.pros_times)
        self.pros = self.pros[sorted_args]
        self.pros_times = self.pros_times[sorted_args]

        # extract mappings from student's id to student's date and features
        self.stu_feat = {}
        self.stu_time = {}
        for stu_id, group in stu.groupby('students_id'):
            group_ar = group.values[:, 1:]
            self.stu_feat[stu_id] = np.array([group_ar[i, 1:] for i in range(group_ar.shape[0])])
            self.stu_time[stu_id] = np.array([group_ar[i, 0] for i in range(group_ar.shape[0])])

        # extract mappings from professional's id to professional's date and features
        self.pro_feat = {}
        self.pro_time = {}
        for pro_id, group in pro.groupby('professionals_id'):
            group_ar = group.values[:, 1:]
            self.pro_feat[pro_id] = np.array([group_ar[i, 1:] for i in range(group_ar.shape[0])])
            self.pro_time[pro_id] = np.array([group_ar[i, 0] for i in range(group_ar.shape[0])])

    def __len__(self):
        return len(self.pos_pairs) // self.batch_size