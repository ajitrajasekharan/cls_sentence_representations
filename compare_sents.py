import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import *
import sys
import numpy as np
import pdb




try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    print("Number of embeddings:",len(embeds_dict))
    return embeds_dict


def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 0
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[count] = term
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

class SeEmbeds:
    def __init__(self, model_path,terms_file,embeds_file):
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=False)
        self.terms_dict = read_terms(terms_file)
        self.embeddings = read_embeddings(embeds_file)
        self.cache = True
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}

    def calc_inner_prod(self,vec1,vec2):
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        vec1 /= n1
        vec2 /= n2
        val = np.inner(vec1,vec2)
        return val

def bucket(cos_dict):
        bucket = {}
        for key in cos_dict:
            val = round(cos_dict[key],1)
            if (val in bucket):
                bucket[val] += 1
            else:
                bucket[val] = 1
        sorted_d = OrderedDict(sorted(bucket.items(), key=lambda kv: kv[0], reverse=True))
        return sorted_d

def find_mean_std(bucket_d):
    total = 0
    val = 0
    for key in bucket_d:
        val += bucket_d[key]*key
        total += bucket_d[key]
    mean = float(val)/total
    std_sum = 0
    for key in bucket_d:
        std_sum = (mean - key)*(mean - key)*bucket_d[key]
    std_val = math.sqrt(std_sum/total)
    return round(mean,2),round(std_val,2)



def full_test(b_embeds,output_file):
    index = 0
    results = {}
    similarity_matrix = np.zeros(len(b_embeds.embeddings)*len(b_embeds.embeddings)).reshape(len(b_embeds.embeddings),len(b_embeds.embeddings))
    for i in range(len(b_embeds.embeddings)):
        for j in range(len(b_embeds.embeddings)):
            score = b_embeds.calc_inner_prod(b_embeds.embeddings[i],b_embeds.embeddings[j])
            results[b_embeds.terms_dict[j]] = score
            similarity_matrix[i][j] = score
        final_sorted_d = OrderedDict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))
        bucket_dict = bucket(final_sorted_d)
        mean,std_val = find_mean_std(bucket_dict)
        print("Cosine distance histogram"," mean:",mean," std:", std_val, " 1-sigma:", round(mean + std_val,2), "2-sigma:", round(mean + 2*std_val,2), " 3-sigma:",round(mean + 3*std_val,2)," 4-sigma:",round(mean + 4*std_val,2), " 5-sigma:",round(mean + 5*std_val,2)," 6-sigma:",round(mean + 6*std_val,2) )
        for k in bucket_dict:
            print(str(k),str(bucket_dict[k]))
        for term in final_sorted_d:
            print(term,round(final_sorted_d[term],1))
        print()
        break
        with open(output_file,"wb") as wfp:
                np.save(wfp,similarity_matrix)




def main():
    if (len(sys.argv) != 4):
        print("Usage: <Bert model path - to load tokenizer>  <sentences file> <vector file>")
    else:
        b_embeds =SeEmbeds(sys.argv[1],sys.argv[2],sys.argv[3])
        full_test(b_embeds,sys.argv[2].rstrip("txt") + "npy")
        #while (True):
        #    print("Enter sentence index")
        #    sent = input()
        #    neigh_test(b_embeds,sent)




if __name__ == '__main__':
    main()
