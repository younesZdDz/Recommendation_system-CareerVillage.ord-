import pandas as pd
import numpy as np
import keras
import os

from sklearn.neighbors import KDTree

from preprocessors.queproc import QueProc
from preprocessors.proproc import ProProc
from utils.utils import TextProcessor

tp = TextProcessor()


class Predictor:
    """
    Class for handling closest professionals or questions queries
    """

    def __init__(self, model: keras.Model, que_data: pd.DataFrame, stu_data: pd.DataFrame, pro_data: pd.DataFrame,
                 que_proc: QueProc, pro_proc: ProProc, que_to_stu: dict, pos_pairs: list):
        """
        :param model: compiled Keras model
        :param que_data: processed questions's data
        :param stu_data: processed student's data
        :param pro_data: processed professional's data
        :param que_proc: question's data processor
        :param pro_proc: professional's data processor
        :param que_to_stu: mappings from question's id to its author id
        :param pos_pairs: list of positive question-student-professional-time pairs
        """
        self.model = model

        # construct mappings from entity id to features
        self.que_dict = {row.values[0]: row.values[2:] for i, row in que_data.iterrows()}
        self.stu_dict = {stu: group.values[-1, 2:] for stu, group in stu_data.groupby('students_id')}
        self.pro_dict = {pro: group.values[-1, 2:] for pro, group in pro_data.groupby('professionals_id')}

        self.entity_to_paired = dict()

        # construct mappings from entity to other entities it was in positive pair
        for que, stu, pro, time in pos_pairs:
            if que not in self.entity_to_paired:
                self.entity_to_paired[que] = {pro}
            else:
                self.entity_to_paired[que].add(pro)

            if pro not in self.entity_to_paired:
                self.entity_to_paired[pro] = {que}
            else:
                self.entity_to_paired[pro].add(que)

        # form final features for 1all known questions and professionals

        que_feat, que_ids, pro_feat, pro_ids = [], [], [], []

        for que in self.que_dict.keys():
            cur_stu = que_to_stu[que]
            if cur_stu in self.stu_dict:
                # actual question's features are both question and student's features
                que_feat.append(np.hstack([self.stu_dict[cur_stu], self.que_dict[que]]))
                que_ids.append(que)

        for pro in self.pro_dict.keys():
            pro_feat.append(self.pro_dict[pro])
            pro_ids.append(pro)

        self.pro_feat = np.vstack(pro_feat)
        self.pro_ids = np.vstack(pro_ids)
        self.que_feat = np.vstack(que_feat)
        self.que_ids = np.vstack(que_ids)

        # create two encoders
        self.que_model = model.que_model
        self.pro_model = model.pro_model

        # compute latent vectors for questions and professionals
        self.que_lat_vecs = self.que_model.predict(self.que_feat)
        self.pro_lat_vecs = self.pro_model.predict(self.pro_feat)

        # create KDTree trees from question and professional latent vectors
        self.que_tree = KDTree(self.que_lat_vecs)
        self.pro_tree = KDTree(self.pro_lat_vecs)

        # initialize preprocessors
        self.que_proc = que_proc
        self.pro_proc = pro_proc

    def __get_que_latent(self, que_df: pd.DataFrame, que_tags: pd.DataFrame) -> np.ndarray:
        """
        Get latent vectors for questions in raw format
        """
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        # extract and preprocess question's features
        que_feat = self.que_proc.transform(que_df, que_tags).values[:, 2:]

        # actual question's features are both question and student's features
        stu_feat = np.vstack([self.stu_dict[stu] for stu in que_df['questions_author_id']])
        que_feat = np.hstack([stu_feat, que_feat])

        # encode question's data to get latent representation
        lat_vecs = self.que_model.predict(que_feat)

        return lat_vecs

    def __get_pro_latent(self, pro_df: pd.DataFrame, que_df: pd.DataFrame, ans_df: pd.DataFrame,
                         pro_tags: pd.DataFrame) -> np.ndarray:
        """
        Get latent vectors for professionals in raw format
        """
        pro_df['professionals_date_joined'] = pd.to_datetime(pro_df['professionals_date_joined'])
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])
        ans_df['answers_date_added'] = pd.to_datetime(ans_df['answers_date_added'])

        # extract and preprocess professional's features
        pro_feat = self.pro_proc.transform(pro_df, que_df, ans_df, pro_tags)

        # select the last available version of professional's features
        pro_feat = pro_feat.groupby('professionals_id').last().values[:, 1:]

        # encode professional's data to get latent representation
        lat_vecs = self.pro_model.predict(pro_feat)

        return lat_vecs

    def __construct_df(self, ids, sims, scores):
        scores = np.round(scores, 4)
        tuples = []
        for i, cur_id in enumerate(ids):
            for j, sim in enumerate(sims[i]):
                if sim[0] not in self.entity_to_paired.get(cur_id, {}):
                    tuples.append((cur_id, sim[0], scores[i, j]))
        score_df = pd.DataFrame(tuples, columns=['id', 'match_id', 'match_score'])
        return score_df

    def __get_ques_by_latent(self, ids: np.ndarray, lat_vecs: np.ndarray, top: int) -> pd.DataFrame:
        """
        Get top questions with most similar latent representations to given vectors
        """
        dists, ques = self.que_tree.query(lat_vecs, k=top)
        ques = self.que_ids[ques]
        scores = np.exp(-dists)
        return self.__construct_df(ids, ques, scores)

    def __get_pros_by_latent(self, ids: np.ndarray, lat_vecs: np.ndarray, top: int) -> pd.DataFrame:
        """
        Get top professionals with most similar latent representations to given vectors
        """
        dists, pros = self.pro_tree.query(lat_vecs, k=top)
        pros = self.pro_ids[pros]
        scores = np.exp(-dists)
        return self.__construct_df(ids, pros, scores)

    def find_pros_by_que(self, que_df: pd.DataFrame, que_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        """
        Get top professionals with most similar internal representation to given questions

        :param que_df: question's data in raw format
        :param que_tags: questions's tags in raw format
        :param top: number of professionals for each question to return
        :return: dataframe of question's ids, matched professional's ids and similarity scores
        """
        lat_vecs = self.__get_que_latent(que_df, que_tags)
        return self.__get_pros_by_latent(que_df['questions_id'].values, lat_vecs, top)