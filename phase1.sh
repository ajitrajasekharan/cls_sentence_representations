set -v
input=${1?"Specify input file"}
final_output=${2?"specify output file"}
model_path=${3-"./model"}
compare_sents=${4-1}
max_seq=${5-512}
output_sent_vector_file=${6-"sent_vectors.npy"}
output_sent_indices_file=${7-"sent_indices.txt"}
echo "Using model: $model_path"
set -v
rm -f *.npy
python cls_embed_gen.py -model $model_path -input $input -output $output_sent_vector_file -output_index $output_sent_indices_file -max_seq $max_seq
ls -l *.npy
sleep 10
if [ $compare_sents -eq 1 ]
then
    inp_file="expanded_$input"
    echo $inp_file
    python compare_sents.py -model $model_path -input $inp_file -vecs $output_sent_vector_file  > $final_output
fi
ls -l *.npy
