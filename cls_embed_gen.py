import sys
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import pdb



class CL_SE:
    def __init__(self, model_path,is_mlm):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.is_mlm = is_mlm
        if (is_mlm):
            self.model = BertForMaskedLM.from_pretrained(model_path)
        else:
            self.model = BertModel.from_pretrained(model_path)
        #self.model = BertModel.from_pretrained(model_path)


    def gen_embedding(self,sent):
        inputs = self.tokenizer(sent, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs[0][1][0][0].tolist() if self.is_mlm else outputs[0][0][0].tolist() # Just the CLS vector



def main(model_path,inp_file,out_file,is_mlm,sent_indices_file):
    se = CL_SE(model_path,is_mlm)
    vecs = []
    lines_fp = open("expanded_" + inp_file,"w")
    with open(inp_file,"r") as fp:
        count = 1
        for line in fp:
            if (len(line) > 0):
                line_len = len(line.split())
                for i in range(line_len):
                        if (i != line_len -1): #generate only full lines. Comment this to create partial lines
                            continue
                        trunc_line = ' '.join(line.split()[:i+1])
                        vec = se.gen_embedding(trunc_line)
                        print(count,trunc_line,len(vec))
                        vecs.append(vec)
                        lines_fp.write(str(count) + ' ' + trunc_line+'\n')
                count += 1

    with open(out_file,"w") as fp:
        fp.write(str(vecs))

    with open(sent_indices_file,"w") as fp:
        for i in range(len(vecs)):
                fp.write(str(i+1) + '\n')





if __name__ == "__main__":
    if (len(sys.argv) != 5):
        print("Usage prog <model path>  <input sentence file> <output vector file name> <output sent indices file name>")
    else:
            main(sys.argv[1],sys.argv[2],sys.argv[3],True,sys.argv[4])
