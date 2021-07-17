import sys
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import pdb
import argparse
import numpy as np


#2 for CLS and SEP
DEFAULT_MAX_SEQUENCE_LENGTH=510
DEFAULT_MODEL_PATH = "./"
DEFAULT_INDEX_FILE="sent_indices.txt"


class CL_SE:
    def __init__(self, model_path,is_mlm,do_lower):
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.is_mlm = is_mlm
        if (is_mlm):
            self.model = BertForMaskedLM.from_pretrained(model_path)
        else:
            self.model = BertModel.from_pretrained(model_path)
        #self.model = BertModel.from_pretrained(model_path)

    def truncate_line(self,line,max_seq):
            line = line.rstrip('\n')
            tokenized_text = self.tokenizer.tokenize('[CLS] ' + line + ' [SEP]')
            max_len = min(max_seq,len(tokenized_text)-1)
            line  = ' '.join(tokenized_text[1:max_len])
            combined_sent = ''
            sent_arr = line.split()
            for i in range(len(sent_arr)):
                if (sent_arr[i][0] == '#' and len(sent_arr[i]) > 2 and sent_arr[i][1] == '#'):
                    combined_sent = combined_sent + sent_arr[i][2:]
                else:
                    combined_sent = combined_sent + ' ' + sent_arr[i]
            line = combined_sent.strip()
            return line


    def gen_embedding(self,sent):
        inputs = self.tokenizer(sent, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs[0][1][0][0].tolist() if self.is_mlm else outputs[0][0][0].tolist() # Just the CLS vector



def main(results):
    model_path = results.model
    inp_file = results.input
    out_file = results.output
    is_mlm = True
    sent_indices_file = results.output_index
    max_seq = results.max_seq
    tolower = results.tolower
    se = CL_SE(model_path,is_mlm,tolower)
    vecs = []
    lines_fp = open("expanded_" + inp_file,"w")
    with open(inp_file,"r") as fp:
        count = 1
        for line in fp:
            if (len(line) > 0):
                line = ' '.join(line.split()[:max_seq])
                line_len = len(line.split())
                for i in range(line_len):
                        if (i != line_len -1): #generate only full lines. Comment this to create partial lines
                            continue
                        trunc_line = ' '.join(line.split()[:i+1])
                        trunc_line = se.truncate_line(trunc_line,max_seq)
                        vec = se.gen_embedding(trunc_line)
                        print(count,trunc_line,len(trunc_line.split()))
                        vecs.append(vec)
                        lines_fp.write(str(count) + '\t' + trunc_line+ '\t' + line +'\n')
                count += 1

    with open(out_file,"wb") as fp:
        np.save(fp,np.array(vecs))

    with open(sent_indices_file,"w") as fp:
        for i in range(len(vecs)):
                fp.write(str(i+1) + '\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentence embeddings for input  ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input", help='Input file with sentences')
    parser.add_argument('-output', action="store", dest="output", help='Output file with embeddings')
    parser.add_argument('-output_index', action="store", dest="output_index",default=DEFAULT_INDEX_FILE, help='Output sentence indices file')
    parser.add_argument('-max_seq', action="store", dest="max_seq",type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH,help=' Max sequence length ')
    parser.add_argument('-tolower', dest="tolower", action='store_true',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-no-tolower', dest="tolower", action='store_false',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.set_defaults(tolower=False)
    results = parser.parse_args()
    main(results)
