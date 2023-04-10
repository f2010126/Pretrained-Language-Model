#!/bin/bash -l
#SBATCH --job-name=pregen_data
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/pregen.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/pregen.error


echo 'Activate Environment'
source ~/tinybert_nlp/bin/activate

: '
echo 'Download and unzip data'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT/data
echo 'Download Wikipedia Data'
wget -v https://github.com/GermanT5/wikipedia2corpus/releases/download/v1.0/dewiki-20220201-clean-part-01
wget -v https://github.com/GermanT5/wikipedia2corpus/releases/download/v1.0/dewiki-20220201-clean-part-02
wget -v https://github.com/GermanT5/wikipedia2corpus/releases/download/v1.0/dewiki-20220201-clean-part-03

echo 'conatenate files'
cat dewiki-20220201-clean-part-* > dewiki-20220201-clean.zip

echo 'unzip files'
unzip dewiki-20220201-clean.zip
'
echo 'Run Pregenerate Data'
cd $(ws_find zap_hpo_og)/TinyBert/TinyBERT
python3 pregenerate_training_data.py --train_corpus 'data/dewiki-20220201-clean.txt' \
                  --bert_model 'bert-base-german-dbmdz-cased' \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 30 \
                  --output_dir 'data/ep_30_pretraining_data' \
