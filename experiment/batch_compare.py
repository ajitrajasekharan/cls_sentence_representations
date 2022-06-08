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
import interactive_gen_vectors as gv


DEFAULT_MODEL_PATH='allenai/scibert_scivocab_cased'
DEFAULT_TO_LOWER=False
DEFAULT_TOP_K = 50
ACCRUE_THRESHOLD = 1

debug = True

import logging



def batch_compare(bert_embeds,topk,input_file,output_file):
    sent_count = 1
    sent_dict = OrderedDict()
    wfp = open(output_file,"w")
    with open(input_file) as fp:
        for line in fp:
            line = line.rstrip()
            ret_dict1 = bert_embeds.get_sent_descs(results.topk,line) 
            sim_val,dummy = bert_embeds.compare_sents(ret_dict1,ret_dict1)
            sent_dict[sent_count] = {"sent":line,"val":sim_val,"dict":ret_dict1}
            print("Processed :",sent_count)
            sent_count += 1
    for i in sent_dict:
        sent_info_i = sent_dict[i]
        ret_dict1 = sent_info_i["dict"]
        norm_val = sent_info_i["val"]
        if (debug):
            print("New Pivot start")
        wfp.write("\n\n\nNew Pivot start\n")
        count = 1
        out_dict = {}
        for j in sent_dict:
            sent_info_j = sent_dict[j]
            ret_dict2 = sent_info_j["dict"]
            sim_val,sim_info = bert_embeds.compare_sents(ret_dict1,ret_dict2)
            sim_val /= norm_val
            sim_val = round(sim_val,4)
            out_dict[count] = {"s_i":sent_info_i["sent"], "s_j": sent_info_j["sent"],"val": sim_val,"sim_details":sim_info}
            count += 1
        sorted_d = OrderedDict(sorted(out_dict.items(), key=lambda kv: kv[1]["val"], reverse=True))
        count = 1
        for i in sorted_d:
            node = sorted_d[i]
            if (debug):
                print(count,node["s_i"],"|",node["s_j"],str(node["val"]))
            wfp.write(str(count) + "] " + node["s_i"] + "|" + node["s_j"] + "|" + str(node["val"]) + "\n")
            wfp.write("SIM_DETAILS:              " +  str(node["sim_details"]) + "\n\n")
            count += 1
        wfp.write("\n")
        wfp.flush()
    wfp.flush()
    wfp.close()
        
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vectors for input sentences',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-topk', action="store", dest="topk", default=DEFAULT_TOP_K,type=int,help='Number of neighbors to use')
    parser.add_argument('-common', action="store", dest="common", default="common_descs.txt",help='Filter common descs')
    parser.add_argument('-embeds', action="store", dest="embeds", default="scibert/bert_vectors.txt",help='Bert learned vector embeddings')
    parser.add_argument('-input', action="store", dest="input", default="sentences.txt",help='File of sentences to compare with each other')
    parser.add_argument('-output', action="store", dest="output", default="output.txt",help='Output of  compare')
    parser.add_argument('-tolower', dest="tolower", action='store_true',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-no-tolower', dest="tolower", action='store_false',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.set_defaults(tolower=False)

    results = parser.parse_args()
    be_embeds = gv.BertSentEmbeds(results.model,results.tolower,results.common,results.embeds)
    batch_compare(be_embeds,results.topk,results.input,results.output)
