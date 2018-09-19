# -*- coding: utf-8 -*-


import json, codecs, os, sys
import numpy as np
import pandas as pd

def read_json(test_json_path, is_answer=False):
    test_json_list = []
    test_data = []
    with codecs.open(test_json_path, 'r', encoding='utf-8') as f:
        while True:
            try:
                 test_json = json.loads(f.readline())
                 # print(test_json)
                 # break
                 test_json_list.append(test_json)
                 if is_answer:
                    test_data.append((test_json['alternatives'], test_json['query_id'], test_json['answer']))
                 else:
                    test_data.append((test_json['alternatives'], test_json['query_id']))
            except:
                break
    if is_answer:         
        test_data = pd.DataFrame(test_data, columns=['alternatives', 'query_id','answer']) 
    else:         
        test_data = pd.DataFrame(test_data, columns=['alternatives', 'query_id'])
    return test_data


if __name__ == '__main__':  

    train_json_path = '../ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'

    test_json_path = '../ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    
    valid_json_path = '../ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    
    # test_data = read_json(test_json_path)
    train_data = read_json(train_json_path,True)
    sys.exit(0)
    valid_data = read_json(valid_json_path,True)
    sum(train_data['alternatives'].map(lambda x: x.split('|')[0]) == train_data['answer'])
    sum(valid_data['alternatives'].map(lambda x: x.split('|')[0]) == valid_data['answer'])

    sys.exit(0)
    with codecs.open('../log/pred.txt','w',encoding='utf-8') as f:
        for rowid, row in test_data.iterrows():
            
            answer = row['alternatives'].split('|')[0]
            print(str(row['query_id']) + '\t' + answer)
            f.write(str(row['query_id']) + '\t' + answer + '\n')