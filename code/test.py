# -*- coding: utf-8 -*-

import json, codecs, os, sys, random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import fool
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import keras.backend as K
from keras.utils import plot_model

import models

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
    
    return word_to_num, word_vec, words

    
class TexttoVec():

    def __init__(self, 
                 words_path='../log/words.npy',
                 word_vec_path='../log/word_vec.npy'):
        
        self.word_to_num, word_vec, self.words = get_vec(words_path,word_vec_path)
        self.word_vec = np.vstack([np.zeros((1, 300)),word_vec])

        self.num_to_word = dict((idx, word) 
                                for word, idx in self.word_to_num.items())
        self.num_to_word[0] = ''
        
        self.to_num = lambda x: self.word_to_num.get(x,0)
        self.to_word = lambda x: self.num_to_word.get(x,'')
        self.num_word = max(self.word_to_num.values())
    
    def text_to_vec(self, text):
        
        words = fool.cut(text)[0]
        
        return list(map(self.to_num, words))
        
    def vec_to_text(self, vec):
        
        # words = fool.cut(text)[0]
        
        return list(map(self.to_word, vec))    
    
def test_word_vec(word_to_num, word_vec):

    word_left = '广州'
    vec_left = word_vec[word_to_num[word_left]].reshape((1,-1))
    for word_right in ['深圳', '上海', '北京', '天津' ,'东京', '本田']:
        vec_right = word_vec[word_to_num[word_right]].reshape((1,-1))
        print('%s → %s' %(word_left, word_right), 
                np.linalg.norm((vec_left-vec_right),axis=1))
    
    word_left = '丰田'
    vec_left = word_vec[word_to_num[word_left]].reshape((1,-1))
    for word_right in ['本田', '宝马', '食品', '手机' ,'牙膏', '猴子']:
        vec_right = word_vec[word_to_num[word_right]].reshape((1,-1))
        print('%s → %s' %(word_left, word_right), 
                np.linalg.norm((vec_left-vec_right),axis=1))
    
    word_left = '不一定'
    vec_left = word_vec[word_to_num[word_left]].reshape((1,-1))
    for word_right in ['也许', '肯定', '知道', '未必' ,'确定', '必然']:
        vec_right = word_vec[word_to_num[word_right]].reshape((1,-1))
        print('%s → %s' %(word_left, word_right), 
                np.linalg.norm((vec_left-vec_right),axis=1))
                
def read_json(json_path, is_answer=False):

    # json_list = []
    data = []
    with codecs.open(json_path, 'r', encoding='utf-8') as f:
        
        for idx, line in enumerate(f.readlines()):
            if idx > 99:
                break
            try:
                json_content = json.loads(line)
            except:
                print(line)
                continue
            # json_list.append(json_content)
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
                    
    return data
    
def answer_to_vec(data, ttv):

    answers_list = []
    label_list = []
    for i in range(3):

        a = data['alternatives'].map(lambda x: x.split('|')[i])
        l = (a == data['answer']).astype(np.int32)
        label_list.append(l.values.reshape((-1,1)))
        answers_list.append(a.map(ttv.text_to_vec))
         
    return answers_list, label_list

def create_data(data, ttv, 
                passage_path='../log/passage.npy',
                query_path='../log/query.npy',
                answer_path='../log/answer.npy',
                label_path='../log/label.npy'):
    
    check_path = [passage_path, query_path, answer_path, label_path]
    
    if np.all([os.path.exists(path) for path in check_path]):
        passage_seq = np.load(passage_path)
        query_seq = np.load(query_path)
        answer_seq = np.load(answer_path)
        y = np.load(label_path)
    else:
    
        answer_vec_list, y_list = answer_to_vec(data, ttv)
        
        query_vec = data['query'].map(ttv.text_to_vec)
        query_seq = pad_sequences(query_vec.tolist(),
                                  maxlen=16,padding='post', truncating='post')
        
        passage_vec = data['passage'].map(ttv.text_to_vec)
        passage_seq = pad_sequences(passage_vec.tolist(),
                                    maxlen=128,padding='post', truncating='post')
        
        answer_seq_list = [pad_sequences(
                            answer_vec_list[i].tolist(), 
                            maxlen=2,
                            padding='post', 
                            truncating='post') for i in range(3)]

    
        # passage_seq = np.vstack([passage_seq]*3)
        # query_seq = np.vstack([query_seq]*3)
        answer_seq = np.hstack(answer_seq_list)
        # y = np.vstack(y_list)
        y = to_categorical(data['label'].values,3)
        # np.save(passage_path.replace('.npy',''), passage_seq)
        # np.save(query_path.replace('.npy',''), query_seq)
        # np.save(answer_path.replace('.npy',''), answer_seq)
        # np.save(label_path.replace('.npy',''), y)
        
    print(passage_seq.shape, query_seq.shape, answer_seq.shape)
        
    return [passage_seq, query_seq, answer_seq], y
    
def shuffle_a(x):
    a = x.split('|')
    random.shuffle(a)
    return '|'.join(a)

def idx_label(x):
    a = x['alternatives'].split('|')
    return a.index(x['answer'])

def find_best_prob(eval_pred, eval_y, prob = 0.5):
    
    idx = eval_pred.max(axis=1) > prob
    acc = np.mean(eval_pred.argmax(axis=1)[idx] == eval_y.argmax(axis=1)[idx])
    print(eval_pred.shape, sum(idx), acc)
    
    
if __name__ == '__main__':  

    from imp import reload
    reload(models)  
    K.clear_session()

    lines = []
    
    ttv = TexttoVec(words_path='../log/words.npy',
                    word_vec_path='../log/word_vec.npy')
                        
    test_word_vec(ttv.word_to_num, ttv.word_vec)
    
    train_json_path = '../ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'

    test_json_path = '../ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    
    valid_json_path = '../ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    
    eval_data = read_json(valid_json_path,True)
    # eval_data = eval_data.sample(frac=1)
    eval_data['label'] = eval_data.loc[:,['alternatives','answer']].apply(idx_label, axis=1)
    
 
    eval_inputs, eval_y = create_data(eval_data, 
                                ttv,passage_path='../log/eval_passage.npy',
                                query_path='../log/eval_query.npy',
                                answer_path='../log/eval_answer.npy',
                                label_path='../log/eval_label.npy')

    model = models.get_model_cnn(eval_inputs, ttv.word_vec)
    
    # model = models.test_embde(train_inputs[0], 
                             # ttv.word_vec)
    # model.predict(train_inputs[0])
    # model.summary()
    plot_model(model, to_file='../log/model.png', show_shapes=True)
    model_path = '../log/model-0.79.h5'
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('Load weighs from', model_path)
    # checkpoint = ModelCheckpoint(
                    # filepath='../log/model-{val_loss:.2f}.h5', 
                    # monitor='val_loss', 
                    # save_best_only=True, 
                    # save_weights_only=True,verbose=1,period=2)
    res = model.evaluate(np.hstack(eval_inputs), eval_y)
    print(res)
    # model.fit(np.hstack(train_inputs), 
              # train_y, 
              # batch_size=128, epochs=1000,callbacks=[checkpoint],
              # validation_data=(np.hstack(valid_inputs), valid_y))
    eval_pred = model.predict(np.hstack(eval_inputs))
    find_best_prob(eval_pred, eval_y, prob = 0.6)
    
    