import sys
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
from collections import OrderedDict
import pdb
import json

OUTPUT_FILE="bias.txt"

def read_descs(file_name):
    ret_dict = {}
    with open(file_name) as fp:
        line = fp.readline().rstrip("\n")
        if (len(line) >= 1):
            ret_dict[line] = 1
        while line:
            line = fp.readline().rstrip("\n")
            if (len(line) >= 1):
                ret_dict[line] = 1
    return ret_dict



class CL:
    def __init__(self, model_path,vocab_file):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.vocab_dict = read_descs(vocab_file)


    def extract(self):
        dummy_sent = "This is a dummy sentence"
        inputs = self.tokenizer(dummy_sent, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs[0][2].tolist()



def main(model_path,vocab_file):
    se = CL(model_path,vocab_file)
    vecs = []
    bias = se.extract()
    assert(len(se.vocab_dict) == len(bias))
    count = 0
    ret_dict = OrderedDict()
    for key in se.vocab_dict:
        ret_dict[key] = bias[count]
        count += 1
    with open(OUTPUT_FILE,"w") as fp:
        for key in ret_dict:
            value_str = "{0} {1:2.6f}".format(key,ret_dict[key])
            print(value_str)
            fp.write(value_str + '\n')
    print("Output :",OUTPUT_FILE," created:",len(ret_dict))






if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage prog <model path> <vocab file>")
    else:
            main(sys.argv[1],sys.argv[2])
