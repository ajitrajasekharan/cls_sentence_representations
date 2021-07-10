import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import BertTokenizer
import sys
import numpy as np
import pdb
import argparse
import time


DEFAULT_MODEL_PATH = "model"
DEFAULT_INPUT_VECTORS = "sent_vectors.npy"
DEFAULT_INPUT="sentence.txt"



try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')




def read_embeddings(embeds_file):
    return np.load(embeds_file)


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
        assert(len(self.terms_dict) == len(self.embeddings))
        self.cache = True
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}


def bucket(cos_dict,log_scale,round_val):
        bucket = {}
        for key in cos_dict:
            if (log_scale):
                val = round(np.log10(cos_dict[key]["score"]),round_val)
            else:
                val = round(cos_dict[key]["score"],round_val)
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

def cache_matrix(b_embeds,matrix_file,normalize):
    print("Computing similarity matrix (takes approx 5 minutes for ~100,000x100,000 matrix ...")
    start = time.time()
    vec_a = b_embeds.embeddings.T #shape (1024,)
    if (normalize):
        print("**Vectors being normalized. Do not do this for BERT vectors. Magnitudes carry information***")
        vec_a = vec_a/np.linalg.norm(vec_a,axis=0) #Note BERT vector magnitudes capture information. So dont normalize them
    #vec_b = vec_a #(1024,)
    vec_a = vec_a.T #(,1024)
    #similarity_matrix = np.dot(vec_a,vec_b) #(,1024) . (1024,)
    similarity_matrix = np.inner(vec_a,vec_a) #(,1024) . (1024,)
    end = time.time()
    time_val = (end-start)*1000
    print("Similarity matrix computation complete.Elapsed:",time_val/(1000*60)," minutes")
    with open(matrix_file,"wb") as wfp:
        np.save(wfp,similarity_matrix)
    return similarity_matrix


def full_test(b_embeds,results):
    matrix_file = results.input.rstrip("txt") + "npy"
    normalize = results.normalize
    index = 0
    try:

        print("Attempting to load similarity matrix...:",matrix_file)
        similarity_matrix = np.load(matrix_file)
        print("Similarity matrix loaded:",similarity_matrix.shape)
    except:
        print("Cached matrix:",matrix_file," not found. Constructing matrix from sentences")
        similarity_matrix = cache_matrix(b_embeds,matrix_file,normalize)
    assert(len(similarity_matrix[0]) == len(b_embeds.terms_dict))

    for i in range(len(similarity_matrix[0])):
        results = {}
        for j in range(len(similarity_matrix[0])):
            score = similarity_matrix[i][j]
            results[j] = {"score":score,"sent":b_embeds.terms_dict[j]}
        final_sorted_d = OrderedDict(sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True))
        bucket_dict = bucket(final_sorted_d,False if normalize else True,1 if normalize else 2)
        mean,std_val = find_mean_std(bucket_dict)
        print("Cosine distance histogram"," mean:",mean," std:", std_val, " 1-sigma:", round(mean + std_val,2), "2-sigma:", round(mean + 2*std_val,2), " 3-sigma:",round(mean + 3*std_val,2)," 4-sigma:",round(mean + 4*std_val,2), " 5-sigma:",round(mean + 5*std_val,2)," 6-sigma:",round(mean + 6*std_val,2),"**Log scale**" if normalize == False else "**Linear scale**" )
        for k in bucket_dict:
            print(str(k),str(bucket_dict[k]))
        print("***Pivot sentence:",b_embeds.terms_dict[i])
        for term in final_sorted_d:
            value = np.log10(final_sorted_d[term]["score"]) if normalize == False else final_sorted_d[term]["score"]
            print(term,final_sorted_d[term]["sent"],round(value,1 if normalize else 2))
        print()
        #break



def main():
    parser = argparse.ArgumentParser(description='Cluster sentences  ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input",default=DEFAULT_INPUT, help='Input file with sentences,assumes no format. But assumption is ***this list matches with input embeddings***')
    parser.add_argument('-normalize', dest="normalize", action='store_true',help='Normalize vectors before dot product. NOT DONE FOR CLS since magnitude carries information')
    parser.add_argument('-no-normalize', dest="normalize", action='store_false',help='Normalize vectors before dot product. NOT DONE FOR CLS since magnitude carries information')
    parser.add_argument('-vecs', action="store", dest="vecs",default=DEFAULT_INPUT_VECTORS, help='Input file with embeddings. This list should match sentences file list')
    parser.set_defaults(normalize=True)
    results = parser.parse_args()
    b_embeds =SeEmbeds(results.model,results.input,results.vecs)
    full_test(b_embeds,results)



if __name__ == '__main__':
    main()

