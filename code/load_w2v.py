# -*- coding: utf-8 -*-


import json, codecs, os, sys
import numpy as np
import pandas as pd
import fool
from keras.preprocessing.sequence import pad_sequences



def get_vec(words_path='../log/words.npy',
            word_vec_path='../log/word_vec.npy'):

    if (not os.path.exists(words_path)) or (not os.path.exists(word_vec_path)):
    
        with codecs.open('../log/sgns.zhihu.word','r',encoding='utf-8') as f:
            print('num of word, dim of vector',f.readline().strip())
            for line in f.readlines():
                lines.append(line.strip().split(' '))
        lines = np.array(lines)
        words = lines[:,0]
        word_vec = lines[:,1:].astype(np.float32)
        word_to_num = dict((word, idx+1) 
                        for idx, word in enumerate(words))
        
        np.save(words_path.replace('.npy',''), words)
        np.save(word_vec_path.replace('.npy',''), word_vec)
    
    else:
        words = np.load(words_path)
        word_to_num = dict((word, idx+1) 
                        for idx, word in enumerate(words))
        word_vec = np.load(word_vec_path)

    return word_to_num, word_vec

    
class TexttoVec():

    def __init__(self, 
                 words_path='../log/words.npy',
                 word_vec_path='../log/word_vec.npy'):
        
        self.word_to_num, self.word_vec = get_vec(words_path,word_vec_path)
        self.num_to_word = dict((idx, word) 
                                for word, idx in self.word_to_num.items())
        self.num_to_word[0] = ''
        
        self.to_num = lambda x: self.word_to_num.get(x,0)
        self.to_char = lambda x: self.num_to_word.get(x,'')
    
    def text_to_vec(self, text):
        
        words = fool.cut(text)[0]
        
        return list(map(self.to_num, words))
        
    
def test_word_vec(word_to_num, word_vec):

    word_left = '广州'
    vec_left = word_vec[word_to_num[word_left]-1].reshape((1,-1))
    for word_right in ['深圳', '上海', '北京', '天津' ,'东京', '本田']:
        vec_right = word_vec[word_to_num[word_right]-1].reshape((1,-1))
        print('%s → %s' %(word_left, word_right), 
                np.linalg.norm((vec_left-vec_right),axis=1))
    
    word_left = '丰田'
    vec_left = word_vec[word_to_num[word_left]-1].reshape((1,-1))
    for word_right in ['本田', '宝马', '食品', '手机' ,'牙膏', '猴子']:
        vec_right = word_vec[word_to_num[word_right]-1].reshape((1,-1))
        print('%s → %s' %(word_left, word_right), 
                np.linalg.norm((vec_left-vec_right),axis=1))
    
    word_left = '不一定'
    vec_left = word_vec[word_to_num[word_left]-1].reshape((1,-1))
    for word_right in ['也许', '肯定', '知道', '未必' ,'确定', '必然']:
        vec_right = word_vec[word_to_num[word_right]-1].reshape((1,-1))
        print('%s → %s' %(word_left, word_right), 
                np.linalg.norm((vec_left-vec_right),axis=1))
                
def read_json(json_path, is_answer=False):

    json_list = []
    data = []
    with codecs.open(json_path, 'r', encoding='utf-8') as f:
        
        for idx, line in enumerate(f.readlines()):
            if idx > 1000:
                break
            try:
                json_content = json.loads(line)
            except:
                print(line)
                continue
            json_list.append(json_content)
            if is_answer:
                data.append((json_content['alternatives'], 
                             json_content['query_id'], 
                             json_content['answer'],
                             json_content['query'],
                             json_content['passage']))
            else:
                data.append((json_content['alternatives'], 
                             json_content['query_id'],
                             json_content['query'],
                             json_content['passage']))
    
    if is_answer:         
        data = pd.DataFrame(data, 
                    columns=['alternatives','query_id','answer','query','passage']) 
    else:         
        data = pd.DataFrame(data, 
                    columns=['alternatives', 'query_id','query','passage'])
                    
    return data, json_list
    
def answer_to_vec(data, ttv):

    answers_list = []
    for i in range(3):
    
        a = data['alternatives'].map(lambda x: x.split('|')[i])
        answers_list.append(a.map(ttv.text_to_vec))
    # answers_list.append(data['alternatives'].map(lambda x: x.split('|')[1]))
    # answers_list.append(data['alternatives'].map(lambda x: x.split('|')[2]))
    
                
    return answers_list
    
    
if __name__ == '__main__':  

    lines = []
    
    ttv = TexttoVec(words_path='../log/words.npy',
                    word_vec_path='../log/word_vec.npy')
                        
    test_word_vec(ttv.word_to_num, ttv.word_vec)
    
    train_json_path = '../ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'

    test_json_path = '../ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    
    valid_json_path = '../ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    
    train_data, train_json_list = read_json(train_json_path,True)
        
    answer_vec_list = answer_to_vec(train_data, ttv)
    
    query_vec = train_data['query'].map(ttv.text_to_vec)
    query_seq = pad_sequences(query_vec.tolist(),
                              maxlen=16,padding='post', truncating='post')
    
    passage_vec = train_data['passage'].map(ttv.text_to_vec)
    passage_seq = pad_sequences(query_vec.tolist(),
                                maxlen=128,padding='post', truncating='post')
    
    answer_seq = pad_sequences(answer_vec_list[0].tolist(),
                               maxlen=2,padding='post', truncating='post')
    
    