import os 
import sys
sys.path.extend(['..'])
import pandas as pd

from utils.utils import TextProcessor
from nlp.doc2vec import pipeline_d2v
from nlp.lda import pipeline_lda

pd.set_option('display.max_columns', 100, 'display.width', 1024)
pd.options.mode.chained_assignment = None

DATA_PATH, SPLIT_DATE, DUMP_PATH = '../data/', '2019-01-01', '../dump/'

if __name__ == '__main__':
    
    tp = TextProcessor()

    # ##################################################################################################################
    #
    #                                                       READ
    #
    # ##################################################################################################################

    answers = pd.read_csv(os.path.join(DATA_PATH, 'answers.csv'), parse_dates=['answers_date_added'])
    answers['answers_body'] = answers['answers_body'].apply(tp.process)
    ans_train = answers[answers['answers_date_added'] < SPLIT_DATE]

    questions = pd.read_csv(os.path.join(DATA_PATH, 'questions.csv'), parse_dates=['questions_date_added'])
    questions['questions_title'] = questions['questions_title'].apply(tp.process)
    questions['questions_body'] = questions['questions_body'].apply(tp.process)
    questions['questions_whole'] = questions['questions_title'] + ' ' + questions['questions_body']
    que_train = questions[questions['questions_date_added'] < SPLIT_DATE]

    professionals = pd.read_csv(os.path.join(DATA_PATH, 'professionals.csv'), parse_dates=['professionals_date_joined'])
    professionals['professionals_headline'] = professionals['professionals_headline'].apply(tp.process)
    professionals['professionals_industry'] = professionals['professionals_industry'].apply(tp.process)
    pro_train = professionals[professionals['professionals_date_joined'] < SPLIT_DATE]

    students = pd.read_csv(os.path.join(DATA_PATH, 'students.csv'), parse_dates=['students_date_joined'])
    stu_train = students[students['students_date_joined'] < SPLIT_DATE]

    tags = pd.read_csv(os.path.join(DATA_PATH, 'tags.csv'))
    tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: tp.process(x, allow_stopwords=True))

    tag_que = pd.read_csv(os.path.join(DATA_PATH, 'tag_questions.csv')) \
        .merge(tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
    tag_pro = pd.read_csv(os.path.join(DATA_PATH, 'tag_users.csv')) \
        .merge(tags, left_on='tag_users_tag_id', right_on='tags_tag_id')

    # ##################################################################################################################
    #
    #                                                       TRAIN
    #
    # ##################################################################################################################

    print('TRAIN')

    # calculate and save tag and industry embeddings on train data
    print('doc2vec: embeddings training')
    tag_embs, ind_embs, head_d2v, ques_d2v = pipeline_d2v(que_train, ans_train, pro_train, tag_que, tag_pro, 10)
    