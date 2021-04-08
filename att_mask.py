import pdb
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from collections import OrderedDict


#outputs[1][0][0][0][0]
def get_att_for_layer(outputs,pivot,tokenized_text,inp_layer):
    ret_dict = {}
    #print(len(outputs[1][0][0]))
    for layer in range(len(outputs[1][0][0])):
        if (layer != inp_layer and inp_layer != -1):
            continue
        curr_layer = {}
        #print("LAYER:",layer)
        for term  in range(len(outputs[1][0][0][layer])):
            if (term != pivot):
                continue
            #print(len(outputs[1][0][0][layer][term]))
            for neigh in range(len(outputs[1][0][0][layer][pivot])):
                #print(tokenized_text[neigh],outputs[1][0][0][layer][pivot][neigh].tolist())
                curr_layer[tokenized_text[neigh]] = outputs[1][0][0][layer][pivot][neigh].tolist()
            sorted_d = OrderedDict(sorted(curr_layer.items(), key=lambda kv: kv[1], reverse=True))
            ret_dict[layer] = sorted_d
            #print(layer,sorted_d)
    return ret_dict

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('./')
model = BertForSequenceClassification.from_pretrained('./')

#text = "since the last month patient exhibited no signs of acute myloid leukemia"
#text = "imatinib mesylate is used to treat cancer"
#text = "her hypophysitis secondary to ipilimumab was well managed with supplemental hormones"
#text = "I would thus be concerned about neuromyelitis optica and also possibly multiple sclerosis"
#text = "John fell down and broke his leg while playing in the garden with Anne"
text = "He went to prison cell with a cell phone to draw blood cell samples from inmates"
#text = "he fell down and broke his leg"
inputs = tokenizer(text, return_tensors="pt")
text = '[CLS]' + text + '[SEP]'
tokenized_text = tokenizer.tokenize(text)
for i in range(len(tokenized_text)):
        print(i,tokenized_text[i],end=' ')
print()
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, output_attentions=True)
print("--------------------")
for i in range(len(tokenized_text)):
        if (i == 0 or i == len(tokenized_text) -1 ):
            continue
        for j in range(len(tokenized_text)):
                print(j,tokenized_text[j],end=' ')
        print("\nAttention for token:",tokenized_text[i])
        att_masks = get_att_for_layer(outputs,i,tokenized_text,11)
        for node in att_masks:
            print(node,att_masks[node])
        pdb.set_trace()
print("--------------------")
