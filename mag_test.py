import sys
from transformers import BertTokenizer, BertModel
import torch
import pdb
import numpy as  np
from collections import OrderedDict



class CL_SE:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)


    def get_mags(self,sent):
        text = '[CLS]' + sent + '[SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(sent, return_tensors="pt")
        outputs = self.model(**inputs)
        count = 0
        print(len(tokenized_text),len(inputs['input_ids'][0]))
        assert(len(tokenized_text) == len(inputs['input_ids'][0]))
        scores_dict = {}
        for vec in outputs[0][0]:
            n1 = np.linalg.norm(vec.tolist())
            print(tokenized_text[count],n1)
            scores_dict[tokenized_text[count] + '_' + str(count)] = n1
            count += 1
        print("------------- Sorted ----")
        final_sorted_d = OrderedDict(sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=False))
        for term in final_sorted_d:
            print(term,final_sorted_d[term])



def main(model_path):
    se = CL_SE(model_path)
    while (True):
        print("Enter sentence index")
        sent = input()
        se.get_mags(sent)





if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage prog <model path>")
    else:
            main(sys.argv[1])
