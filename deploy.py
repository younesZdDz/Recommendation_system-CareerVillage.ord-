import sys
import os 
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..') )

import nltk
nltk.download('stopwords')

from flask import request
from flask import Flask
from flask import render_template

import json

import pandas as pd

import pickle
from datetime import datetime

from utils.utils import TextProcessor
from models.distance import DistanceModel
from predictor.predictor import Predictor, Formatter
from preprocessors.queproc import QueProc
from preprocessors.proproc import ProProc
import traceback
import time
start_time = time.time()

pd.set_option('display.max_columns', 100, 'display.width', 1024)

# Set oath to data
DATA_PATH = 'data'
SAMPLE_PATH = 'demo_data'
DUMP_PATH = 'dump'



# init flask server
app = Flask(__name__, static_url_path='', template_folder='view')

# Routes
print("--- %s seconds ---" % (time.time() - start_time))

@app.route('/')
def index():
  return render_template('index.html')


@app.route("/api/question", methods = ['POST'])
def question():
    try:
      que_dict = {
          'questions_id': ['0'],
          'questions_author_id': [],
          'questions_date_added': [str(datetime.now())],
          'questions_title': [],
          'questions_body': [],
          'questions_tags': []
      }

      data = request.get_json()

      for key, val in data.items():
        if key in que_dict and val:
          que_dict[key].append(str(val))


      for key, val in que_dict.items():
        if not val:
           return json.dumps([], default=str)

      que_df, que_tags = Formatter.convert_que_dict(que_dict)
      tmp = pred.find_ques_by_que(que_df, que_tags)
      final_df = formatter.get_que(tmp).fillna('')
      final_data = final_df.to_dict('records')

      return json.dumps(final_data, allow_nan=False) 

    except Exception as e:
      traceback.print_exc()
      return json.dumps([], default=str)



@app.route("/api/professional", methods = ['POST'])
def professional():
  try:
    pro_dict = {
        'professionals_id': [],
        'professionals_location': [],
        'professionals_industry': [],
        'professionals_headline': [],
        'professionals_date_joined': [],
        'professionals_subscribed_tags': []
      }

    data = request.get_json()

    pro = professionals_sample[professionals_sample['professionals_id'] == data['professionals_id']]
    pro = pro.to_dict('records')[0]

    tag = pro_tags_sample[pro_tags_sample['tag_users_user_id'] == data['professionals_id']]

    for key, val in pro.items():
        if key in pro_dict and val:
          pro_dict[key].append(str(val))
    
    pro_dict['professionals_subscribed_tags'].append(' '.join(list(tag['tags_tag_name'])))    
    
    for key, val in pro_dict.items():
      if not val:          
         return json.dumps([], default=str)
    
    pro_df, pro_tags = Formatter.convert_pro_dict(pro_dict)
    tmp = pred.find_ques_by_pro(pro_df, questions, answers, pro_tags)
    final_df = formatter.get_que(tmp).fillna('')
    
    final_data = final_df.to_dict('records')
    
    return json.dumps(final_data, allow_nan=False) 
      
  except Exception as e:
    traceback.print_exc()
    return json.dumps([], default=str)

