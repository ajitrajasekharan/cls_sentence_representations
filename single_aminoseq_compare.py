import sys
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import pdb
import argparse
import cls_embed_gen as cls
import compare_sents as Cmp
from collections import OrderedDict


DEFAULT_MAX_SEQUENCE_LENGTH=510
DEFAULT_MODEL_PATH = "model/"
DEFAULT_SPLIT_NGRAM = 3


def get_neighs(comp,vec):
    results = {}
    for i in range(len(comp.embeddings)):
        score = comp.calc_inner_prod(comp.embeddings[i],vec)
        results[comp.terms_dict[i]] = score
    final_sorted_d = OrderedDict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))
    return final_sorted_d


def split_input(ngram,line):
    pdb.set_trace()
    chunks = [line[i:i+ngram] for i in range(0, len(line), ngram)]
    return ' '.join(chunks)


def main(results):
    model_path = results.model
    inp_file = results.input
    out_file = results.output
    is_mlm = True
    max_seq = results.max_seq
    ref_vecs = results.ref_vecs
    ref_input = results.ref_input
    tolower = results.tolower
    ngram = results.ngram
    se = cls.CL_SE(model_path,is_mlm,tolower)
    comp_module = Cmp.SeEmbeds(model_path,ref_input,ref_vecs)
    with open(inp_file,"r") as fp:
        count = 1
        for line in fp:
            if (len(line) > 0):
                if (ngram != 0):
                    line = split_input(ngram,line.rstrip('\n'))
                line = se.truncate_line(line,max_seq)
                vec = se.gen_embedding(line)
                final_sorted_d = get_neighs(comp_module,vec)
                input_data = line
                break #Just a single line in this test
    with open(out_file,"w") as wfp:
        wfp.write("Input:" + input_data + '\n')
        for term in final_sorted_d:
            #print(term,round(final_sorted_d[term],3))
            wfp.write(term + ' ' +  str(round(final_sorted_d[term],3)) + '\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentence embeddings for input  ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input", help='Input file with sentences')
    parser.add_argument('-ref_input', action="store", dest="ref_input", help='Input file with reference sentences')
    parser.add_argument('-ref_vecs', action="store", dest="ref_vecs", help='Input file with reference sentence vectors')
    parser.add_argument('-output', action="store", dest="output", help='Output file of neighbors')
    parser.add_argument('-max_seq', action="store", dest="max_seq",type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH,help=' Max sequence length ')
    parser.add_argument('-ngram', action="store", dest="ngram",type=int, default=DEFAULT_SPLIT_NGRAM,help=' Default splitting of input ')
    parser.add_argument('-tolower', dest="tolower", action='store_true',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-no-tolower', dest="tolower", action='store_false',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.set_defaults(tolower=False)
    results = parser.parse_args()
    main(results)
