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
import random


SINGLETONS_TAG  = "_singletons_ "
ZSCORE = 2
MAX_PICK_PERCENT = .05

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    return embeds_dict


def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict


class SentEmbeds:
    def __init__(self, terms_file,embeds_file):
        cache_embeds = True
        normalize = True
        self.terms_dict = read_terms(terms_file)
        self.embeddings = read_embeddings(embeds_file)
        self.cache = cache_embeds
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}
        self.dist_zero_cache = {}
        self.normalize = normalize



    def adaptive_gen_pivot_graphs(self):
        count = 1
        total = len(self.terms_dict)
        picked_dict = OrderedDict()
        pivots_dict = OrderedDict()
        singletons_arr = []
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open("sent_cluster_pivots.txt","w")
        max_pick_count = len(self.terms_dict)*MAX_PICK_PERCENT
        for key in self.terms_dict:
            count += 1
            #print(":",key)
            if (key in picked_dict):
                continue
            print("Processing ",count," of ",total)
            picked_dict[key] = 1
            temp_sorted_d,dummy = self.get_distribution_for_term(key)
            z_threshold = self.find_zscore(temp_sorted_d,ZSCORE) # this is not a normal distribution - yet using asssuming it is for a reasonable thresholding
            tail_len,threshold = self.get_tail_length(key,temp_sorted_d,z_threshold,max_pick_count)
            sorted_d = self.get_terms_above_threshold(key,threshold)
            arr = []
            for k in sorted_d:
                picked_dict[k] = 1
                arr.append(k)
            if (len(arr) > 1):
                max_mean_term,max_mean, std_dev,s_dict = self.find_pivot_subgraph(arr)
                if (max_mean_term not in pivots_dict):
                    new_key  = max_mean_term
                else:
                    print("****Term already a pivot node:",max_mean_term, "key  is :",key)
                    new_key  = max_mean_term + "++" + key
                pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"std_dev":std_dev,"terms":arr}
                print(new_key,max_mean,std_dev,arr)
                dfp.write(new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
            else:
                if (len(arr) == 1):
                    print("***Singleton arr for term:",key)
                    singletons_arr.append(key)
                else:
                    print("***Empty arr for term:",key)
                    pdb.set_trace()
                    assert(0)
                    empty_arr.append(key)

        dfp.write(SINGLETONS_TAG + str(singletons_arr) + "\n")
        pivots_dict[SINGLETONS_TAG] = {"key":SINGLETONS_TAG,"orig":SINGLETONS_TAG,"mean":0,"std_dev":0,"terms":singletons_arr}
        with open("pivots.json","w") as fp:
            fp.write(json.dumps(pivots_dict))
        dfp.close()
        inv_pivots_dict = OrderedDict()
        count = 1
        cluster_sizes = 0
        for key in pivots_dict:
            arr = pivots_dict[key]["terms"]
            arr_size = len(arr)
            for term in arr:
                if (int(term) not in inv_pivots_dict):
                    inv_pivots_dict[int(term)] = []
                else:
                    print("Already present:",term)
                inv_pivots_dict[int(term)].append({"index":count,"size":arr_size})
            count += 1
            if (key != SINGLETONS_TAG):
                cluster_sizes += arr_size
        avg_cluster_size = cluster_sizes/float(count - 2) #not counting singleton 
        sorted_d = OrderedDict(sorted(inv_pivots_dict.items(), key=lambda kv: kv[0], reverse=False))
        with open("inv_pivots.json","w") as fp:
            fp.write(json.dumps(sorted_d))
        dfp.close()
        cluster_stats_dict = {}
        for key in inv_pivots_dict:
            arr_size = len(inv_pivots_dict[key])
            if (arr_size not in cluster_stats_dict):
                cluster_stats_dict[arr_size] = 1
            else:
                cluster_stats_dict[arr_size] += 1
        sorted_d = OrderedDict(sorted(cluster_stats_dict.items(), key=lambda kv: kv[0], reverse=False))
        final_dict = {"avg_cluster_size":round(avg_cluster_size,0),"element_inclusion_hist":sorted_d,"singleton_counts":len(pivots_dict[SINGLETONS_TAG]["terms"]),"total_clusters":len(pivots_dict)-1,"total_input":total}
        with open("cluster_stats.json","w") as fp:
            fp.write(json.dumps(final_dict))
        dfp.close()


    def get_tail_length(self,key,sorted_d,threshold,max_pick_count):
        rev_sorted_d = OrderedDict(sorted(sorted_d.items(), key=lambda kv: kv[0], reverse=True))
        count = 0
        pick_count = 0
        cosine_value = 0
        for k in rev_sorted_d:
            if (k >= threshold):
               if (count + sorted_d[k] > max_pick_count):
                    break
               count += sorted_d[k]
               cosine_value = k
            else:
                break
        return count,cosine_value

        
    def find_zscore(self,sorted_d,z_val):
        sum_val = 0
        count = 0
        for k in sorted_d:
            sum_val += k*sorted_d[k]
            count += sorted_d[k]
        mean = sum_val/count
        std_val = 0
        for k in sorted_d:
            std_val += (mean - k)*(mean - k)*sorted_d[k]
        std_dev = math.sqrt(std_val/count)
        print("mean:",mean,"std",std_dev,"threshold",mean + z_val*std_dev)
        return mean + z_val*std_dev
        
            



    def gen_dist_for_vocabs(self,sample_input):
        is_rand = True
        count = 1
        picked_count = 0
        skip_count = 0
        cum_dict = OrderedDict()
        cum_dict_count = OrderedDict()
        zero_dict = OrderedDict()
        tail_lengths = OrderedDict()
        total_tail_length = 0
        if (sample_input):
                sample =  100 - len(self.terms_dict)*.03
        else:
                sample = 100
        max_pick_count = len(self.terms_dict)*MAX_PICK_PERCENT
        for key in self.terms_dict:
            if (is_rand):
                val = random.randint(0,100)
                if (sample_input and val < sample): # this is a biased skip to do a fast cum dist check (3% sample ~ 1000)
                    skip_count+= 1
                    print("Processed:",picked_count,"Skipped:",skip_count,end='\r')
                    continue
            picked_count += 1
            sorted_d,dummy = self.get_distribution_for_term(key)
            print("Processing ",picked_count," of ",len(self.terms_dict))
            threshold = self.find_zscore(sorted_d,ZSCORE) # this is not a normal distribution - yet using asssuming it is for a reasonable thresholding
            tail_len,dummy = self.get_tail_length(key,sorted_d,threshold,max_pick_count)
            tail_lengths[key] = tail_len
            print("tail length:",tail_len)
            total_tail_length += tail_len
            for k in sorted_d:
                val = round(float(k),1)
                #print(str(val)+","+str(sorted_d[k]))
                if (val == 0):
                    zero_dict[key] = sorted_d[k]
                if (val in cum_dict):
                    cum_dict[val] += sorted_d[k]
                    cum_dict_count[val] += 1 # this is not a normal distribution - yet using asssuming it is for a reasonable thresholding
                else:
                    cum_dict[val] = sorted_d[k]
                    cum_dict_count[val] = 1
        for k in cum_dict:
            cum_dict[k] = round(float(cum_dict[k])/cum_dict_count[k],0)
        final_sorted_d = OrderedDict(sorted(cum_dict.items(), key=lambda kv: kv[0], reverse=False))
        print("\nTotal picked:",picked_count)
        with open("cum_dist.txt","w") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            for k in final_sorted_d:
                print(k,final_sorted_d[k])
                p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                fp.write(p_str)

        with open("zero_vec_counts.txt","w",encoding="utf-8") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            final_sorted_d = OrderedDict(sorted(zero_dict.items(), key=lambda kv: kv[1], reverse=True))
            try:
                for k in final_sorted_d:
                    #print(k,final_sorted_d[k])
                    p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                    fp.write(p_str)
            except:
                print("Exception 1")

        with open("tail_counts.txt","w",encoding="utf-8") as fp:
            fp.write("Total picked:" + str(picked_count) + " Average tail len: " + str(round(float(total_tail_length)/picked_count,1)) +  "\n")
            final_sorted_d = OrderedDict(sorted(tail_lengths.items(), key=lambda kv: kv[1], reverse=True))
            try:
                for k in final_sorted_d:
                    #print(k,final_sorted_d[k])
                    p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                    fp.write(p_str)
            except:
                print("Exception 2")



    def get_embedding(self,text):
        vec =  self.get_vector([int(text)-1])
        if (self.cache):
                self.embeds_cache[text] = vec
        return vec


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        #pdb.set_trace()
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        #sq_sum = 0
        #for i in range(len(vec)):
        #    sq_sum += vec[i]*vec[i]
        return vec

    def calc_inner_prod(self,text1,text2):
        if (self.cache and text1 in self.cosine_cache and text2 in self.cosine_cache[text1]):
            return self.cosine_cache[text1][text2]
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        if (vec1 is None or vec2 is None):
            #print("Warning: at least one of the vectors is None for terms",text1,text2)
            return 0
        val = np.inner(vec1,vec2)
        if (self.cache):
            if (text1 not in self.cosine_cache):
                self.cosine_cache[text1] = {}
            self.cosine_cache[text1][text2] = val
        return val

    def get_distribution_for_term(self,term1):
        if (term1 in self.dist_threshold_cache):
            return self.dist_threshold_cache[term1],self.dist_zero_cache
        terms_count = self.terms_dict
        dist_dict = {}
        val_dict = {}
        zero_dict = {}
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2)

            val = round(val,2)
            if (val in dist_dict):
                dist_dict[val] += 1
            else:
                dist_dict[val] = 1
            val = round(val,1)
            if (val >= -.05 and val <= .05):
                zero_dict[term2] = 0
        sorted_d = OrderedDict(sorted(dist_dict.items(), key=lambda kv: kv[0], reverse=False))
        self.dist_threshold_cache[term1] = sorted_d
        self.dist_zero_cache = zero_dict
        return sorted_d,zero_dict

    def get_terms_above_threshold(self,term1,threshold):
        final_dict = {}
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2)
            val = round(val,2)
            if (val >= threshold):
                final_dict[term2] = val
        sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
        return sorted_d



    #given n terms, find the mean of the connection strengths of subgraphs considering each term as pivot.
    #return the mean of max strength term subgraph
    def find_pivot_subgraph(self,terms):
        max_mean = 0
        std_dev = 0
        max_mean_term = None
        means_dict = {}
        if (len(terms) == 1):
            return terms[0],1,0,{terms[0]:1}
        for i in terms:
            full_score = 0
            count = 0
            full_dict = {}
            for j in terms:
                if (i != j):
                    val = self.calc_inner_prod(i,j)
                    #print(i+"-"+j,val)
                    full_score += val
                    full_dict[count] = val
                    count += 1
            if (len(full_dict) > 0):
                mean  =  float(full_score)/len(full_dict)
                means_dict[i] = mean
                #print(i,mean)
                if (mean > max_mean):
                    #print("MAX MEAN:",i)
                    max_mean_term = i
                    max_mean = mean
                    std_dev = 0
                    for k in full_dict:
                        std_dev +=  (full_dict[k] - mean)*(full_dict[k] - mean)
                    std_dev = math.sqrt(std_dev/len(full_dict))
                    #print("MEAN:",i,mean,std_dev)
        #print("MAX MEAN TERM:",max_mean_term)
        sorted_d = OrderedDict(sorted(means_dict.items(), key=lambda kv: kv[1], reverse=True))
        return max_mean_term,round(max_mean,2),round(std_dev,2),sorted_d







def main():
    if (len(sys.argv) != 3):
        print("Usage:  <sentence [index] file> <vector file> ")
    else:
        b_embeds =SentEmbeds(sys.argv[1],sys.argv[2])
        display_threshold = .4
        while (True):
            print("Enter test type (0-gen cum dist for sentnces; 1-generate clusters ; q to quit")
            val = input()
            if (val == "0"):
                try:
                    b_embeds.gen_dist_for_vocabs(False)
                except:
                    print("Trapped exception")
                sys.exit(-1)
            elif (val == "1"):
                b_embeds.adaptive_gen_pivot_graphs()
                sys.exit(-1)
            elif (val == 'q'):
                sys.exit(-1)
            else:
                print("invalid option")




if __name__ == '__main__':
    main()
