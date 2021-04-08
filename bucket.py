import sys,pdb
from collections import OrderedDict

def bucket(input_file):
    with open(input_file) as fp:
        bucket = {}
        for line in fp:
            if (len(line) > 1):
                    val = round(float(line.strip()),2)
                    #print(val)
                    if (val in bucket):
                        bucket[val] += 1
                    else:
                        bucket[val] = 1
        sorted_d = OrderedDict(sorted(bucket.items(), key=lambda kv: kv[0], reverse=True))
        for k in sorted_d:
            print(k,sorted_d[k])



if __name__ == "__main__":
        if (len(sys.argv) == 2):
            bucket(sys.argv[1])
        else:
            print("Enter input file with numbers")

