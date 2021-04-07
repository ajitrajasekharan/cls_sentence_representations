# cls_sentence_representations

Sentence representation using the [CLS] vector of a pre-trained model without fine-tuning. 

# Installation


Setup pytorch environment with/without GPU support using link https://github.com/ajitrajasekharan/multi_gpu_test

pip install gdown

./fetch_model.sh

# Usage

*phase1.sh test.txt output.txt*

This will output the neighbors for each sentence in test.txt
Example below


![DES](DES.png)

*phase2.sh*
this can be used to either examine the sentence vectors or create clusters. The stats of the clusters are also output

![stats](stats.png)

# Note. 
This may require a code patch to transformer file modeling_bery.py in order to work
![patch](patch.png)


# License
MIT License
