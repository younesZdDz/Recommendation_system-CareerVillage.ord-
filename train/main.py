import os 
import sys
sys.path.extend(['..'])
import pandas as pd

from utils.utils import TextProcessor
from NLP.doc2vec import pipeline_d2v
from NLP.lda import pipeline_lda
from preprocessors.queproc import QueProc
from preprocessors.stuproc import StuProc
from preprocessors.proproc import ProProc
from train.generator import BatchGenerator

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
    print('lda: topic model training')
    lda_dic, lda_tfidf, lda_model = pipeline_lda(que_train, 10)

    # extract and preprocess feature for all three main entities
    print('processor: questions')
    que_proc = QueProc(tag_embs, ques_d2v, lda_dic, lda_tfidf, lda_model)
    que_data = que_proc.transform(que_train, tag_que)

    print('processor: students')
    stu_proc = StuProc()
    stu_data = stu_proc.transform(stu_train, que_train, ans_train)

    print('processor: professionals')
    pro_proc = ProProc(tag_embs, ind_embs, head_d2v, ques_d2v)
    pro_data = pro_proc.transform(pro_train, que_train, ans_train, tag_pro)
    # ##################################################################################################################
    #
    #                                                       INGESTION
    #
    # ##################################################################################################################

    print('INGESTION')

    # construct dataframe used to extract positive pairs
    pairs_df = questions.merge(answers, left_on='questions_id', right_on='answers_question_id') \
        .merge(professionals, left_on='answers_author_id', right_on='professionals_id') \
        .merge(students, left_on='questions_author_id', right_on='students_id')

    pairs_df = pairs_df[['questions_id', 'students_id', 'professionals_id', 'answers_date_added']]

    # extract positive pairs
    pos_pairs = list(pairs_df.loc[pairs_df['answers_date_added'] < SPLIT_DATE].itertuples(index=False, name=None))

    # mappings from professional's id to his registration date. Used in batch generator
    pro_to_date = {row['professionals_id']: row['professionals_date_joined'] for i, row in professionals.iterrows()}

    bg = BatchGenerator(que_data, stu_data, pro_data, 64, pos_pairs, pos_pairs, pro_to_date)