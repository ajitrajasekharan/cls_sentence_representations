set -v
input=${1?"Specify input file"}
final_output=${2?"specify output file"}
output_sent_vector_file=${3-"sent_vectors.txt"}
output_sent_indices_file=${3-"sent_indices.txt"}
model_path=${4-"./model"}
echo "Using model: $model_path"
set -v
python cls_embed_gen.py $model_path $input $output_sent_vector_file $output_sent_indices_file
inp_file="expanded_$input"
echo $inp_file
python compare_sents.py $model_path $inp_file $output_sent_vector_file  > $final_output
