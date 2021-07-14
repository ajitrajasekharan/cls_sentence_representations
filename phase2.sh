input=${1?"specify input text file used in phase1"}
python sentence_dist.py -terms $input -vectors sent_vectors.npy -zscore 4

