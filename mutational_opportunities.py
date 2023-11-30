import os
import gzip
import numpy as np
from collections import defaultdict
import pandas as pd

complement = defaultdict(lambda: "N")
complement.update({'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'})
nucleotides = list(sorted(complement.keys()))
codontable = defaultdict(lambda: "-")
codontable.update({
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': 'X', 'TAG': 'X',
    'TGC': 'C', 'TGT': 'C', 'TGA': 'X', 'TGG': 'W', '---': '-'})


def open_nuc_matrix(path):
    df_nuc = pd.read_csv(path, sep="\t")
    return {name: q for name, q in zip(df_nuc["Name"], df_nuc["Rate"])}


def open_fasta(path):
    print(f"Loading fasta file {path}...")
    outfile = {}
    ali_file = gzip.open(path, 'rt')
    seq_name = ""
    for line in ali_file:
        if line.startswith(">"):
            seq_name = line[1:].strip()
            outfile[seq_name] = []
        else:
            seq = line.strip().upper()
            outfile[seq_name].append(seq)
    ali_file.close()
    print(f"Fasta file {path} loaded.")
    return {name: "".join(seq) for name, seq in outfile.items()}


def group_by_gene(fasta_cds):
    dico_genes = defaultdict(list)
    for seq_name, seq in fasta_cds.items():
        gene_name = seq_name[seq_name.find("[gene=") + 6:]
        gene_name = gene_name[:gene_name.find("]")]
        dico_genes[gene_name].append(seq)
    return dico_genes


def build_codon_neighbors():
    codon_neighbors = defaultdict(list)
    for ref_codon, ref_aa in codontable.items():
        if ref_aa == "-" or ref_aa == "X":
            continue
        for frame, ref_nuc in enumerate(ref_codon):
            for alt_nuc in [nuc for nuc in nucleotides if nuc != ref_nuc]:
                alt_codon = ref_codon[:frame] + alt_nuc + ref_codon[frame + 1:]
                alt_aa = codontable[alt_codon]
                if alt_aa != 'X':
                    syn = alt_aa == ref_aa
                    codon_neighbors[ref_codon].append((syn, ref_nuc, alt_nuc, alt_codon, alt_aa))
    return codon_neighbors


def mutational_opportunities(seq, codon_neighbors, mutation_matrix):
    mu_syn, mu_nonsyn, n_total = 0.0, 0.0, 0.0
    for c_site in range(len(seq) // 3):
        ref_codon = seq[c_site * 3:c_site * 3 + 3]
        ref_aa = codontable[ref_codon]
        if ref_aa == "X" or ref_aa == "-":
            continue

        n_total += 1

        for (syn, ref_nuc, alt_nuc, alt_codon, alt_aa) in codon_neighbors[ref_codon]:
            mutation_rate = mutation_matrix[f"q_{ref_nuc}_{alt_nuc}"]
            assert np.isfinite(mutation_rate)
            if syn:
                mu_syn += mutation_rate
            else:
                mu_nonsyn += mutation_rate
    return mu_syn, mu_nonsyn, n_total


def main():
    output_path = "results/mutational_opportunities.tsv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dico_opp = defaultdict(list)
    codon_neighbors = build_codon_neighbors()

    fasta_path = "data/GCF_000699145.1_ASM69914v1_cds_from_genomic.fna.gz"
    gene_stats = "data/geneStatsExp.tsv"
    nuc_matrix = "data/nucmat/b10k.tsv"
    error_path = "results/aa_seq.fasta"

    mutation_matrix = open_nuc_matrix(nuc_matrix)
    df_gene_stats = pd.read_csv(gene_stats, sep="\t")
    set_genes = {row["gene"]: row for _, row in df_gene_stats.iterrows()}
    fasta_cds = open_fasta(fasta_path)
    fasta_groups = group_by_gene(fasta_cds)
    nb_groups_genes = {k: len(v) for k, v in fasta_groups.items()}
    avg_nb_groups_genes = np.mean(list(nb_groups_genes.values()))
    print(f"Average number of CDS per group: {avg_nb_groups_genes:.2f}")

    intersection = set_genes.keys() & fasta_groups.keys()
    print(f"Number of genes: {len(set_genes)}")
    print(f"Number of fasta genes: {len(fasta_groups)}")
    print(f"Number of intersection genes: {len(intersection)}")
    assert len(intersection) == len(set_genes)

    error_file = open(error_path, "w")
    dico_count = defaultdict(int)
    for gene_name in set_genes:
        dico_gene_opp = defaultdict(list)
        for seq in fasta_groups[gene_name]:
            if len(seq) % 3 != 0:
                dico_count["not_divisible_by_3"] += 1
                error_file.write(f">{gene_name}\n{seq}\n")
                aa_seq = "".join([codontable[seq[i:i + 3]] for i in range(0, len(seq), 3)])
                error_file.write(f">{gene_name}_aa\n{aa_seq}\n")
            else:
                dico_count["ok"] += 1
            mu_syn, mu_nonsyn, n_total = mutational_opportunities(seq, codon_neighbors, mutation_matrix)
            dico_gene_opp["mu_syn"].append(mu_syn)
            dico_gene_opp["mu_nonsyn"].append(mu_nonsyn)
            dico_gene_opp["mu_total"].append(mu_syn + mu_nonsyn)
            dico_gene_opp["n_total"].append(n_total)

        dico_opp["gene"].append(gene_name)
        for k, v in dico_gene_opp.items():
            dico_opp[k].append(np.mean(v))
        ratio_array = np.array(dico_gene_opp["mu_syn"]) / np.array(dico_gene_opp["mu_nonsyn"])
        dico_opp["ratio"].append(np.mean(ratio_array))

    error_file.close()
    df = pd.DataFrame(dico_opp)
    df.to_csv(output_path, index=False, sep="\t")
    count_total = sum(dico_count.values())
    for k, v in dico_count.items():
        print(f"{k}: {v} ({v / count_total * 100:.2f}%)")


if __name__ == '__main__':
    main()
