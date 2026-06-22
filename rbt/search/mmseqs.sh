
MMSEQS=[place-your-own-mmseqs-path]
# MMSEQS=[your_mmseqs_path]
input_path=rbt/search/wetlab_seqs.fasta

designed_seqs=rbt/outputs/petase/all_seq_token.fasta

output_path="${input_path%.fasta}_mmseqs_hit.tsv"
echo "output to $output_path"
$MMSEQS easy-search $input_path $designed_seqs $output_path /tmp \
      --min-seq-id 0.5 \
      -c 0.8 \
      --cov-mode 0 \
      --format-output query,target,pident,fident,nident,alnlen,qcov,tcov \
      --threads 96