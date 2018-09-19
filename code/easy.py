# -*- coding: utf-8 -*-


import json, codecs
import numpy as np
import pandas as pd

if __name__ == '__main__':  


    test_json_path = '../ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    
    test_json_list = []
    test_data = []
    with codecs.open(test_json_path, 'r', encoding='utf-8') as f:
        while True:
            try:
                 test_json = json.loads(f.readline())
                 test_json_list.append(test_json)
                 test_data.append((test_json['alternatives'], test_json['query_id']))
            except:
                break
         
    test_data = pd.DataFrame(test_data, columns=['alternatives', 'query_id'])    
    
    with codecs.open('../log/pred.txt','w',encoding='utf-8') as f:
        for rowid, row in test_data.iterrows():
            
            answer = row['alternatives'].split('|')[0]
            print(str(row['query_id']) + '\t' + answer)
            f.write(str(row['query_id']) + '\t' + answer + '\n')