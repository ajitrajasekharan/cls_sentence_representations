import torch
from transformers import *
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import string
import json
import numpy as np
import time


debug = True


import logging

#DEFAULT_MODEL_PATH='bert-large-cased'
#DEFAULT_MODEL_PATH='bert-large-cased' #works best for names
#DEFAULT_MODEL_PATH='bert-base-uncased'
#DEFAULT_MODEL_PATH='./model'
#DEFAULT_MODEL_PATH='./'
DEFAULT_MODEL_PATH='allenai/scibert_scivocab_cased'
DEFAULT_TO_LOWER=False
DEFAULT_TOP_K = 50
ACCRUE_THRESHOLD = 1

def read_embeddings(embeds_file):
    print("Reading  vector embeddings...")
    with open(embeds_file) as fp:
        embeds_list = json.loads(fp.read())
    arr = np.array(embeds_list)
    print("Read :",len(arr), " vector embeddings")
    return arr

def get_sent(to_lower):
    print("Enter sentence (optionally including the special token [MASK] if tolower is set to False). Type q to quit")
    sent = input()
    if (sent == 'q'):
        return sent
    else:
        return  sent.lower() if to_lower else sent

def read_descs(file_name):
    ret_dict = {}
    with open(file_name) as fp:
        line = fp.readline().rstrip("\n")
        if (len(line) >= 1):
            ret_dict[line] = 1
        while line:
            line = fp.readline().rstrip("\n").lower()
            if (len(line) >= 1 and line not in ret_dict):
                ret_dict[line] = 1
    print("Read ",len(ret_dict), " common descs for filtering")
    return ret_dict



class BertSentEmbeds:
    def __init__(self, model_path,to_lower,common_descs,embeds_file):
        logging.basicConfig(level=logging.INFO)
        print("******* MODEL[path] is:",model_path," lower casing is set to:",to_lower)
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        self.common_descs = read_descs(common_descs)
        self.embeddings = read_embeddings(embeds_file)
        self.similarity_matrix = self.cache_matrix(True)

    def cache_matrix(self,normalize):
        b_embeds = self
        print("Computing similarity matrix (takes approx 5 minutes for ~100,000x100,000 matrix ...)")
        start = time.time()
        #pdb.set_trace()
        vec_a = b_embeds.embeddings.T #vec_a shape (1024,)
        if (normalize):
            vec_a = vec_a/np.linalg.norm(vec_a,axis=0) #Norm is along axis 0 - rows
            vec_a = vec_a.T #vec_a shape becomes (,1024)
            similarity_matrix = np.inner(vec_a,vec_a)
        end = time.time()
        time_val = (end-start)*1000
        print("Similarity matrix computation complete.Elapsed:",time_val/(1000*60)," minutes")
        return similarity_matrix

    def get_sent_descs(self,top_k,text):
        text = '[CLS] ' + text + ' [SEP]' 
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        #print(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])


        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            for i in range(len(tokenized_text)):
                    #if (i != 0 and i != len(tokenized_text) - 1):
                    #    continue
                    results_dict = {}
                    masked_index = i
                    for j in range(len(predictions[0][0][masked_index])):
                        tok = self.tokenizer.convert_ids_to_tokens([j])[0]
                        results_dict[tok] = {"id": j, "val": float(predictions[0][0][masked_index][j].tolist())}
                    k = 0
                    sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1]["val"], reverse=True))
                    ret_dict = OrderedDict()
                    if (debug):
                        print("********* Top predictions for token: ",tokenized_text[i])
                    for index in sorted_d:
                        #if (index in string.punctuation or index.startswith('##') or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                        if (index in string.punctuation  or index.lower() in self.common_descs or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                            continue
                        if (debug):
                            print(k+1,index,round(float(sorted_d[index]["val"]),4))
                        ret_dict[index] = {"id": sorted_d[index]["id"], "val": round(float(sorted_d[index]["val"]),4)}
                        k += 1
                        if (k >= top_k):
                            break
                    break
            return ret_dict

    def normalize_vals(self,ret_dict):
        total_val = 0
        for key in ret_dict:
            total_val += ret_dict[key]['val']
        for key in ret_dict:
            ret_dict[key]['val'] = round(ret_dict[key]['val']/total_val,4)
        
            
    def compare_sents(self,ret_dict1,ret_dict2):
        sim_val = 0
        negative_inf = -100000 
        count = 0
        max_vals_dict = {}
        #self.normalize_vals(ret_dict1)
        #self.normalize_vals(ret_dict2)
        for key_i in ret_dict1:
            node_i = ret_dict1[key_i]
            max_val = negative_inf
            max_key = None
            for key_j in ret_dict2:
                node_j = ret_dict2[key_j]
                #print(key_i,key_j,self.similarity_matrix[node_i['id']][node_j['id']])
                curr_val = self.similarity_matrix[node_i['id']][node_j['id']]*node_i['val']*node_j['val']
                if (curr_val > max_val):
                    max_val = curr_val
                    max_key = key_j
                count += 1
            sim_val += max_val
            max_vals_dict[ str(key_i) + ":" + str(max_key)] = max_val
            if (debug):
                print(key_i,max_key,max_val)
        #sim_val = sim_val/len(ret_dict1)
        sim_val = round(sim_val,4)
        if (debug):
            print("Similarity measure:",sim_val)
        sorted_d = OrderedDict(sorted(max_vals_dict.items(), key=lambda kv: kv[1], reverse=True))
        return sim_val,sorted_d
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vectors for input sentences',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-topk', action="store", dest="topk", default=DEFAULT_TOP_K,type=int,help='Number of neighbors to use')
    parser.add_argument('-common', action="store", dest="common", default="common_descs.txt",help='Filter common descs')
    parser.add_argument('-embeds', action="store", dest="embeds", default="scibert/bert_vectors.txt",help='Bert learned vector embeddings')
    parser.add_argument('-tolower', dest="tolower", action='store_true',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-no-tolower', dest="tolower", action='store_false',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.set_defaults(tolower=False)

    results = parser.parse_args()
    try:
        bert_sent_embeds = BertSentEmbeds(results.model,results.tolower,results.common,results.embeds)
        print("To lower casing is set to:",results.tolower)
        val = input("Enter test type: 1 - single test. 2 - pair test:")
        #val = '2'
        if (val == '1'):
            while (True):
                text = get_sent(results.tolower)
                if (text == "q"):
                    print("Quitting")
                    break
                ret_dict1 = bert_sent_embeds.get_sent_descs(results.topk,text)
        else:
            while (True):
                text1 = get_sent(results.tolower)
                #text1 = "The patient felt good today."
                if (text1 == "q"):
                    print("Quitting")
                    break
                ret_dict1 = bert_sent_embeds.get_sent_descs(results.topk,text1)
                text2 = get_sent(results.tolower)
                #text2 = "The patient feels quite well."
                ret_dict2 = bert_sent_embeds.get_sent_descs(results.topk,text2)
                bert_sent_embeds.compare_sents(ret_dict1,ret_dict2)
                
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
