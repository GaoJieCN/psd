# ver = 2023.01.28
import json 
import os 
import numpy as np
import re
import inspect
global t

def nice_print(str_list, data_list, title=None):
    if title != None: print(title)
    for strx, data in zip(str_list, data_list):
        print(strx, ':', data)

t = 0
def debug_write(x):
    global t
    t += 1
    if t == 1:
        with open('../../debug.txt', 'w') as f:
            pass
    assert type(x) == dict
    with open('debug.txt', 'a+') as f:
        x = json.dumps(x) + '\n'
        f.write(x)


def stop():
    stop

def nice_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dump_json_to_dir(dictx, name, dir):
    if not os.path.exists(dir):
        os.makedirs(dir) 
    with open(os.path.join(dir, name), 'w', encoding='utf-8') as f:
        json.dump(dictx, f, ensure_ascii=False, indent=4)

def dump_json(dictx, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        if indent == 4:
            json.dump(dictx, f, ensure_ascii=False, indent=4)
        else: 
            json.dump(dictx, f, ensure_ascii=False) 

def load_json(path):
    try: 
        with open(path, 'r', encoding='utf-8') as f:
            load_dict = json.load(f)
            return load_dict
    except: 
        fffffffffff 

import pprint
# indent：定义几个空格的缩进
def formatprint(data):
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(data)
    
def pair_print(list_tok, list_id=None, lenx=None, name=None, padding_token=None):
    if name != None:
        print(name) 
    
    if type(list_id) == np.ndarray:
        list_id = list_id.tolist() 
    if list_id != None:
        assert len(list_id) == len(list_tok)
    else:
        list_id = list(_ for _ in range(lenx))
    
    if padding_token != None: 
        for i in range(len(list_tok)-1, 0-1, -1):
            if list_tok[i] != padding_token: 
                break 
        i += 1 
        list_tok = list_tok[:i] 
        list_id = list_id[:i] 
    for id, t in zip(list_id, list_tok):
        print('[{}]{} '.format(id, t), end='')
    print('\n')

def get_current_time_str():
    from datetime import datetime
    now = datetime.now()
    strnow = datetime.strftime(now, '%Y-%m-%d-%H-%M-%S')
    return strnow

def _sort_dict(dictx, key='key'):
    if key == 'key':
        return sorted(dictx.items(), key = lambda kv:(kv[0], kv[1]))  
    elif key == 'value':
        return sorted(dictx.items(), key = lambda kv:(kv[1], kv[0]))  
    else:
        raise ValueError('sort key error')

def sort_dict_and_norm(dictx):
    dist = sort_dict(dictx, key='value') 
    sumx = sum([y for x,y in dist]) 
    dist2 = [] 
    for x, y in dist:
        dist2.append((x, round(y/sumx, 3)))  
    return dist2 

def watch_tensor(tensor, name=''):
    if len(tensor.shape) > 2:
        raise ValueError('the shape of tensor should be <= 2d')
    import numpy as np 
    import pandas as pd 
    a = tensor.detach().cpu().numpy().round(3)  
    data = pd.DataFrame(a) 
    data.to_excel('watch_tensor_{}.xlsx'.format(name))  

# src是长字符串 tgt是需要找的子串
def find_all(src, tgt): 
    ret = [] 
    bias = 0 
    while True: 
        index = src.find(tgt)
        if index == -1:
            return ret 
        ret.append(index + bias) 
        src = src[index + len(tgt): ]
        bias += index + len(tgt) 
    return ret

import pickle as pkl 
def load_pkl(path): 
    with open(path, 'rb') as f:
        return pkl.load(f)
    
def dump_pkl(dictx, path): 
    with open(path, 'wb') as f:
        pkl.dump(dictx, f) 

from transformers import AutoTokenizer
def my_tokenizer(model_name): 
    return AutoTokenizer.from_pretrained(model_name) 
     
    