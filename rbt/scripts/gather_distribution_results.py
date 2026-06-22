import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gather distribution results')
    parser.add_argument('res_dir')
    args = parser.parse_args()

    res_files = [f for f in os.listdir(args.res_dir) if f.startswith('sequence_rank') and f.endswith('.txt')]
    _res_files = [f for f in os.listdir(args.res_dir) if f.startswith('res_rank') and f.endswith('.tsv')]
    if res_files == [] or _res_files == []:
        raise ValueError("No sequence_rank files found in the directory")

    res = []
    for f in res_files:
        with open(os.path.join(args.res_dir, f)) as f:
            res.extend(f.readlines())
    # move res_files to rank_files/ folder
    os.makedirs(os.path.join(args.res_dir, 'rank_files'), exist_ok=True)
    for f in res_files:
        os.rename(os.path.join(args.res_dir, f), os.path.join(args.res_dir, 'rank_files', f))
    for f in _res_files:
        os.rename(os.path.join(args.res_dir, f), os.path.join(args.res_dir, 'rank_files', f))

    res = list(set(res)) # remove duplicates
    print("Number of unique sequences: ", len(res))
    with open(os.path.join(args.res_dir, "all_seq_token.fasta"), 'w') as f:
        # f.write(''.join(res))
        for idx, seq in enumerate(res):
            f.write(f">Gather_idx_{idx}\n")
            f.write(f"{seq.strip()}\n")