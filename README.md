# Sentence representation using [CLS] vectors and vector clustering

Sentence representation using the [CLS] vector of a pre-trained model without fine-tuning. 

This repository also contains code for clustering vectors. Input to clustering is two files - vector file (text file) and corresponding descriptor file naming the vectors. Additional paramters are zscore and max tail pick to control how much to pick from distribution tails

# Installation


Setup pytorch environment with/without GPU support using link https://github.com/ajitrajasekharan/multi_gpu_test

pip install gdown

./fetch_model.sh

# Usage

*phase1.sh test.txt output.txt*

This will output the neighbors for each sentence in test.txt
Example below


![DES](DES.png)

Output files

*1) cum_dist.txt:Cumulative histogram of distribution
2) zero_vec_counts.txt: orthogonal vector count
3) tail_counts.txt: tail count of vectors*

*phase2.sh*
this can be used to either examine the sentence vectors or create clusters. The stats of the clusters are also output

![stats](stats.png)

Output files
*1) sent_cluster_pivots.txt:Sentence clusters*
*2) pivots.json: Pivots of clusters*
*3) inv_pivots.json: Inverted pivots*
*4) cluster_stats.json cluster stats*

# Note. 
This may require a code patch to transformer file modeling_bert.py in order to work
![patch](patch.png)


# License
MIT License
