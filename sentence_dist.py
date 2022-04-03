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
import random
import argparse
import time
from sklearn.metrics.pairwise import euclidean_distances


COSINE_DISTANCE = 0
EUCLIDEAN_DISTANCE = 1


ROUNDING_PRECISION=6


SINGLETONS_TAG  = "_singletons_ "
EMPTY_TAG  = "_empty_ "
DEFAULT_ZSCORE = 2 # Two standard deviations away should skip over 95% of mass. For normal disttribution. 1 std - 68% 2 std - 95%; 3 std - 99.7% - 4 std - 99.9937%; 5 std - 99.99937%; 6 std - 99.9999998%
DEFAULT_MAX_PICK_PERCENT = .05 #this is an upper bound to include at most 5 % from tail. This is to cover the case the input is not normally distributed and the tail has large mass



OUTPUT_SENT_CLUSTER_PIVOTS = "sent_cluster_pivots.txt"
OUTPUT_PIVOTS = "pivots.json"
OUTPUT_INVERTED_PIVOTS = "inv_pivots.json"
OUTPUT_CLUSTER_STATS = "cluster_stats.json"
OUTPUT_CUM_DIST = "cum_dist.txt"
OUTPUT_ZERO_VEC_COUNTS = "zero_vec_counts.txt"
OUTPUT_TAIL_COUNTS = "tail_counts.txt"
DESC_CLUSTERS = "desc_clusters.txt"
SUBSERVIENT_CLUSTERS = "subservient_clusters.txt"
EMPTY_SENTENCES = "empty_sentences.txt"
SINGLETON_SENTENCES = "singleton_sentences.txt"

TOP_NEIGHS = "top_neighs.txt"

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')




def read_embeddings(embeds_file):
    #return np.load(embeds_file,allow_pickle=True)
    return np.load(embeds_file)


def read_desc_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[count] = term
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

def create_terms(total_len):
    terms_dict = OrderedDict()
    for i in range(total_len):
        terms_dict[str(i+1)] = i+1 #this is an str for just not braeking previous code that assumes it is str
    print("count of tokens :", len(terms_dict))
    return terms_dict



class SentEmbeds:
    def __init__(self, terms_file,embeds_file,zscore,max_pick,distance):
        cache_embeds = True
        normalize = True
        self.embeddings = read_embeddings(embeds_file)
        self.terms_dict = create_terms(len(self.embeddings))
        self.desc_dict = read_desc_terms(terms_file)
        assert(len(self.desc_dict) == len(self.terms_dict))
        self.cache = cache_embeds
        self.dist_threshold_cache = {}
        self.dist_zero_cache = {}
        self.normalize = normalize
        self.zscore = zscore
        self.max_pick = max_pick
        self.distance = distance
        self.similarity_matrix = self.cache_matrix(True) if distance == COSINE_DISTANCE else  self.cache_euclidean()
        print(f"Distance measure is {distance} 0 - cosine; 1 - euclidean")


    def cache_euclidean(self):
        b_embeds = self
        print("Computing similarity matrix (takes approx 5 minutes for ~100,000x100,000 matrix ...")
        start = time.time()
        vec_a = b_embeds.embeddings #shape (1024,)
        similarity_matrix = euclidean_distances(vec_a, vec_a)
        end = time.time()
        time_val = (end-start)*1000
        print("Similarity matrix computation complete.Elapsed:",time_val/(1000*60)," minutes")
        return similarity_matrix

    def cache_matrix(self,normalize):
        b_embeds = self
        print("Computing similarity matrix (takes approx 5 minutes for ~100,000x100,000 matrix ...")
        start = time.time()
        vec_a = b_embeds.embeddings.T #shape (1024,)
        #vec_a = b_embeds.embeddings #shape (,1024)
        if (normalize):
            vec_a = vec_a/np.linalg.norm(vec_a,axis=0) #Norm is along axis 0 - rows.
            #vec_b = vec_a #(1024,)
            vec_a = vec_a.T #(,1024)
            #similarity_matrix = np.dot(vec_a,vec_b) #(,1024) . (1024,)
            similarity_matrix = np.inner(vec_a,vec_a) #(,1024) . (1024,)
        end = time.time()
        time_val = (end-start)*1000
        print("Similarity matrix computation complete.Elapsed:",time_val/(1000*60)," minutes")
        return similarity_matrix

    def output_desc(self,fp, new_key,key,max_mean,std_dev,arr):
        fp.write("\nPivot: " + self.desc_dict[int(key)] +"\n")
        fp.write(new_key+" "+key+" "+max_mean+" "+ std_dev +"\n")
        element = new_key.split("++")[0]
        if (element != key):
            fp.write("new pivot sentence: " + self.desc_dict[int(element)]+"\n")
        fp.write("pivot sentence: " + self.desc_dict[int(key)]+"\n")
        fp.write("cluster::" + "\n")
        for i in range(len(arr)):
            fp.write(self.desc_dict[int(arr[i])] + "\n")
        fp.write("\n")


    def top_neighs(self):
        count = 0
        total = len(self.terms_dict)
        #neighfp = open(TOP_NEIGHS,"w")

        for key in self.terms_dict:
            count += 1
            #print(":",key)
            print("Processing ",count," of ",total)
            threshold = -1
            sorted_d = self.get_terms_above_threshold(key,threshold)
            neighfp = open("neighs/" + key +".txt","w")
            for key in sorted_d:
                ret_str = f"{self.desc_dict[int(key)]} , {sorted_d[key]:.6f}"
                print(ret_str)
                neighfp.write(ret_str + "\n")
            neighfp.close()
            #neighfp.write("\n\n")
        neighfp.close()


    def adaptive_gen_pivot_graphs(self):
        count = 0
        total = len(self.terms_dict)
        picked_dict = OrderedDict()
        pivots_dict = OrderedDict()
        singletons_arr = []
        fall_through_zscore = .5
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open(OUTPUT_SENT_CLUSTER_PIVOTS,"w")
        descfp = open(DESC_CLUSTERS,"w")
        emptyfp = open(EMPTY_SENTENCES,"w")
        singletonfp = open(SINGLETON_SENTENCES,"w")
        max_pick_count = len(self.terms_dict)*self.max_pick
        for key in self.terms_dict:
            count += 1
            #print(":",key)
            if (key in picked_dict):
                continue
            print("Processing ",count," of ",total)
            picked_dict[key] = 1
            temp_sorted_d,dummy = self.get_distribution_for_term(key)
            z_threshold = self.find_zscore(temp_sorted_d,self.zscore) # this is not a normal distribution (it is a skewed normal distribution) - yet using asssuming it is for a reasonable thresholding
                                                                 # the variance largely captures the right side tail which is no less than the left side tail for BERT models. 
                                                                 # This assumption could be inaccurate for other cases.
                                                                 # We could choose z scores conservatively based on the kind of clusters we want.
            #if (z_threshold > 1):
            #    z_threshold = fall_through_zscore
            #    print("Zscore > 1. Resetting it to:",round(fall_through_zscore,2))
                 
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
                pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"std_dev":std_dev,"size":len(arr),"terms":arr,"subservient":0}
                print(new_key,max_mean,std_dev,arr)
                for k in sorted_d:
                    if (k in pivots_dict and k != max_mean_term and k != key):
                        #pdb.set_trace()
                        print("Marking a pivot list subservient, because it is a child of another pivot \"key\"")
                        pivots_dict[k]["subservient"] = 1
                        iter_dict = dict(pivots_dict) #clone for iter, not sure why I did this! Earlier I planned to delete
                        for j in iter_dict:
                            elements = j.split("++")[0]
                            if (elements == k):
                                print("Also marking  another cluster this pivot was a head of as subservient")
                                pivots_dict[j]["subservient"] = 1
                print("Non singleton cluster:",key,"new_key:",key)
                dfp.write(new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " + str(len(arr)) +  " " +str(arr)+"\n")
                self.output_desc(descfp, new_key,key,str(max_mean),str(std_dev),arr)
            else:
                if (len(arr) == 1):
                    print("***Singleton arr for term:",key)
                    singletons_arr.append(key)
                    singletonfp.write(self.desc_dict[int(key)] +"\n")
                else:
                    print("***Empty arr for term:",key)
                    #pdb.set_trace()
                    #assert(0)
                    #if (fall_through_zscore > .2):
                    #    fall_through_zscore -= .1
                    empty_arr.append(key)
                    emptyfp.write(self.desc_dict[int(key)] +"\n")

        dfp.write(SINGLETONS_TAG + " " + str(len(singletons_arr)) + " " +   str(singletons_arr) + "\n")
        pivots_dict[SINGLETONS_TAG] = {"key":SINGLETONS_TAG,"orig":SINGLETONS_TAG,"mean":0,"std_dev":0,"size":len(singletons_arr),"terms":singletons_arr,"subservient":0}
        dfp.write(EMPTY_TAG + " " + str(len(empty_arr)) + " "   + str(empty_arr) + "\n")
        pivots_dict[EMPTY_TAG] = {"key":EMPTY_TAG,"orig":EMPTY_TAG,"mean":0,"std_dev":0,"size":len(empty_arr),"terms":empty_arr,"subservient": 0}
        with open(OUTPUT_PIVOTS,"w") as fp:
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
            if (key != SINGLETONS_TAG and key not in EMPTY_TAG):
                cluster_sizes += arr_size
        avg_cluster_size = cluster_sizes/float(count - 2) #not counting singleton and empty
        sorted_d = OrderedDict(sorted(inv_pivots_dict.items(), key=lambda kv: kv[0], reverse=False))
        with open(OUTPUT_INVERTED_PIVOTS,"w") as fp:
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
        final_dict = {"avg_cluster_size":round(avg_cluster_size,0),"element_inclusion_hist":sorted_d,"singleton_counts":len(pivots_dict[SINGLETONS_TAG]["terms"]),"empty_counts":len(empty_arr),"total_clusters":len(pivots_dict)-2,"total_input":total,"zcore":self.zscore,"max_pick":self.max_pick} #not counting empty and singletons  in pivots list
        with open(OUTPUT_CLUSTER_STATS,"w") as fp:
            fp.write(json.dumps(final_dict) + "\n")
        with open(SUBSERVIENT_CLUSTERS,"w") as fp:
            for k in pivots_dict:
                if (pivots_dict[k]["subservient"] == 1):
                    element = k.split("++")[0]
                    fp.write(str(k) + " " + self.desc_dict[int(element)] + "\n")

        dfp.close()
        descfp.close()
        emptyfp.close()
        singletonfp.close()

        print("Created output files\n1) {0}:Sentence clusters\n2) {1}: Pivots of clusters\n3) {2}: Inverted pivots\n4) {3} cluster stats\n5) {4} descriptive clusters".format( OUTPUT_SENT_CLUSTER_PIVOTS, OUTPUT_PIVOTS, OUTPUT_INVERTED_PIVOTS, OUTPUT_CLUSTER_STATS,DESC_CLUSTERS))


    def get_tail_length(self,key,sorted_d,threshold,max_pick_count):
        rev_sorted_d = OrderedDict(sorted(sorted_d.items(), key=lambda kv: kv[0], reverse=True))
        count = 0
        pick_count = 0
        cosine_value = 1.1
        for k in rev_sorted_d:
            if (k >= threshold):
               if (count + sorted_d[k] >= max_pick_count): #We pick from the tail only if the new amount be added still keeps it within max_pick_count
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
        print("mean:",round(mean,2),"std",round(std_dev,2),"threshold",round(mean + z_val*std_dev,2))
        return mean + z_val*std_dev





    def gen_dist_for_vocabs(self,sample_percent):
        count = 1
        picked_count = 0
        skip_count = 0
        cum_dict = OrderedDict()
        cum_dict_count = OrderedDict()
        zero_dict = OrderedDict()
        tail_lengths = OrderedDict()
        total_tail_length = 0
        sample =  100 - sample_percent
        max_pick_count = len(self.terms_dict)*self.max_pick
        for key in self.terms_dict:
            val = random.randint(0,100)
            if (val < sample): # this is a biased skip to do a fast cum dist check
                skip_count+= 1
                print("Processed:",picked_count,"Skipped:",skip_count,end='\r')
                continue
            picked_count += 1
            sorted_d,dummy = self.get_distribution_for_term(key)
            print("\nProcessing ",picked_count," of ",len(self.terms_dict))
            threshold = self.find_zscore(sorted_d,self.zscore) # this is not a normal distribution - yet using asssuming it is for a reasonable thresholding
            tail_len,dummy = self.get_tail_length(key,sorted_d,threshold,max_pick_count)
            tail_lengths[key] = tail_len
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
        total = 0;
        for k in cum_dict:
            total  += cum_dict[k]
        for k in cum_dict:
            print(cum_dict[k])
        for k in cum_dict:
            cum_dict[k] = round(float(cum_dict[k])/total,2)
        final_sorted_d = OrderedDict(sorted(cum_dict.items(), key=lambda kv: kv[0], reverse=False))
        print("\nTotal picked:",picked_count)
        with open(OUTPUT_CUM_DIST,"w") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            for k in final_sorted_d:
                print(k,final_sorted_d[k])
                p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                fp.write(p_str)

        with open(OUTPUT_ZERO_VEC_COUNTS,"w",encoding="utf-8") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            fp.write("Total zero vecs count:" + str(len(zero_dict)) + "\n")
            final_sorted_d = OrderedDict(sorted(zero_dict.items(), key=lambda kv: kv[1], reverse=True))
            try:
                for k in final_sorted_d:
                    #print(k,final_sorted_d[k])
                    p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                    fp.write(p_str)
            except Exception as inst:
                print("Exception 1",inst)

        with open(OUTPUT_TAIL_COUNTS,"w",encoding="utf-8") as fp:
            fp.write("Total picked:" + str(picked_count) + " Average tail len: " + str(round(float(total_tail_length)/picked_count,1)) +  "\n")
            final_sorted_d = OrderedDict(sorted(tail_lengths.items(), key=lambda kv: kv[1], reverse=True))
            try:
                for k in final_sorted_d:
                    #print(k,final_sorted_d[k])
                    p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                    fp.write(p_str)
            except Exception as inst:
                print("Exception 2:",inst)
        print("Created output files\n1) {0}:Cumulative histogram of distribution\n2) {1}: orthogonal vector count\n3) {2}: tail count of vectors".format(OUTPUT_CUM_DIST, OUTPUT_ZERO_VEC_COUNTS, OUTPUT_TAIL_COUNTS))



    def calc_distance(self,text1,text2):
        index1 = int(text1) - 1
        index2 = int(text2) -1
        return self.similarity_matrix[index1][index2]

    def get_distribution_for_term(self,term1):
        if (term1 in self.dist_threshold_cache):
            return self.dist_threshold_cache[term1],self.dist_zero_cache
        terms_count = self.terms_dict
        dist_dict = {}
        val_dict = {}
        zero_dict = {}
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_distance(term1,term2)

            val = round(val,2)
            if (val > .3 and val != 1):
                pdb.set_trace()
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
            val = self.calc_distance(term1,term2)
            val = round(val,ROUNDING_PRECISION)
            if (val >= threshold):
                final_dict[term2] = val
        sorted_order = True if self.distance == COSINE_DISTANCE else False
        sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=sorted_order))
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
                    val = self.calc_distance(i,j)
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
    parser = argparse.ArgumentParser(description='Clusters vectors - given input vector file and a corresponding index file. Index file could be terms/words/descriptors of the input vectors.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-vectors', action="store", dest="vectors",required=True, help='file containing vectors - one line per vector')
    parser.add_argument('-terms', action="store", dest="terms",required=True, help='file with sentences/terms desciribng  the  vectors')
    parser.add_argument('-zscore', dest="zscore", action='store',type=float,default=DEFAULT_ZSCORE, help='Minimum standard deviations from mean to consider for clustering')
    parser.add_argument('-max_pick', dest="max_pick", action='store',type=float,default=DEFAULT_MAX_PICK_PERCENT, help='Bound the cluster size maximum')
    parser.add_argument('-distance', action="store", dest="distance",default=EUCLIDEAN_DISTANCE,type=int, help='Distance measure')

    results = parser.parse_args()
    print(results)
    b_embeds =SentEmbeds(results.terms,results.vectors,results.zscore,results.max_pick,results.distance)
    while (True):
        print("Enter test type (0-gen cum dist for sentences; 1-generate clusters ; 2-pick top k neighs q to quit")
        val = input()
        if (val == "0"):
            try:
                print("Enter sample percent. Enter integer value [0-100] 1 for 100 % 1 for  1% sampling. Min 0")
                sample = float(input())
                b_embeds.gen_dist_for_vocabs(sample)
            except Exception as inst:
                print("Trapped exception:",inst)
            sys.exit(-1)
        elif (val == "1"):
            b_embeds.adaptive_gen_pivot_graphs()
            sys.exit(-1)
        elif (val == "2"):
            b_embeds.top_neighs()
            sys.exit(-1)
        elif (val == 'q'):
            sys.exit(-1)
        else:
            print("invalid option")




if __name__ == '__main__':
    main()
