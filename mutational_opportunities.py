import os
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
    ali_file = open(path, 'r')
    seq_name = ""
    for line in ali_file:
        if line.startswith(">"):
            seq_name = line[1:].split(" ")[0]
            outfile[seq_name] = []
        else:
            seq = line.strip().upper()
            outfile[seq_name].append(seq)
    ali_file.close()
    print(f"Fasta file {path} loaded.")
    return {name: "".join(seq) for name, seq in outfile.items()}


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


class Cds(object):
    def __init__(self, chromosome, strand, name):
        """
        :chromosome : (String) Chromosome number (can also be X/Y or Z/W).
        :strand : (String) Strand on which the CDS is encoded.
        :name : (String) Name of the CDS.
        :exons : (List of 2-tuple) List of exons. Each exon is defined by a tuple (start, end),
                 where 'start' and 'end' are in absolute position in the chromosome.
        """
        self.chromosome = chromosome
        assert (strand == "+" or strand == "-")
        self.strand = strand
        self.name = name
        self.exons = []
        self.seq = ""

    def add_exon(self, start_exon, end_exon, fasta_chr):
        start_exon, end_exon = int(start_exon), int(end_exon)
        if start_exon <= end_exon:
            if self.strand == "+":
                self.exons.append((start_exon, end_exon))
                self.seq += fasta_chr[self.chromosome][start_exon - 1:end_exon]
            else:
                self.exons.insert(0, (start_exon, end_exon))
                seq_complement = "".join([complement[nt] for nt in fasta_chr[self.chromosome][start_exon - 1:end_exon]])
                self.seq += seq_complement[::-1]

    def exons_length(self):
        return [j - i + 1 for i, j in self.exons]

    def seq_length(self):
        sum_exons = sum(self.exons_length())
        assert sum_exons == len(self.seq)
        return sum_exons

    def not_coding(self):
        return self.seq_length() % 3 != 0

    def overlapping(self):
        for i in range(len(self.exons) - 1):
            if not self.exons[i][1] < self.exons[i + 1][0]:
                return True
        return False

    def befile_lines(self, chr_prefix):
        lines = []
        for start, end in self.exons:
            lines.append("{0}{1}\t{2}\t{3}\t{4}\t0\t{5}\n".format(chr_prefix, self.chromosome, start, end, self.name,
                                                                  self.strand))
        return lines


def build_dict_cds(file_path, fasta_chr, set_genes):
    print('Loading GTF file...')
    gtf_file = open(file_path, 'r')
    dico_cds = dict()

    for gtf_line in gtf_file:
        if gtf_line.startswith('#'):
            continue

        chromosome, source, feature, start, end, score, strand, frame, comments = gtf_line.replace('\n', '').split('\t')
        if feature != 'CDS':
            continue

        gene_find = comments.find('gene=')
        if gene_find == -1:
            continue
        gene_name = comments[gene_find + 5:].split(";")[0]
        if gene_name not in set_genes:
            continue

        protein_id = comments.find('protein_id=')
        if protein_id == -1:
            protein_name = gene_name
        else:
            protein_name = comments[protein_id + 11:].split(";")[0].strip()
            assert ";" not in protein_name

        if gene_name not in dico_cds:
            dico_cds[gene_name] = {}
        if protein_name not in dico_cds[gene_name]:
            dico_cds[gene_name][protein_name] = Cds(chromosome, strand, protein_name)

        dico_cds[gene_name][protein_name].add_exon(start, end, fasta_chr)

    output_dico_cds = {}
    count, count_not, count_total = 0, 0, 0
    for name, cds_dico in dico_cds.items():
        if len(cds_dico) == 1:
            cds = list(cds_dico.values())[0]
        else:
            cds_name = ""
            for protein_name, cds_p in cds_dico.items():
                cds_p_len = cds_p.seq_length()
                cds_set_len = set_genes[name]["CDSlen"] + len(cds_p.exons)
                if cds_p_len == cds_set_len:
                    cds_name = protein_name
            if cds_name == "":
                count_not += 1
                print(f"Warning: {name} cannot find gene.")
                cds_name = list(cds_dico.keys())[0]
            count += 1
            cds = cds_dico[cds_name]
        count_total += 1
        output_dico_cds[name] = cds
        if cds.not_coding():
            print(f"Warning: {name} is not a multiple of 3.")
        if cds.overlapping():
            print(f"Warning: {name} is overlapping.")
            print(f"Warning: {name} length is {cds.seq_length()} instead of {set_genes[name]}.")
        # if cds.seq_length() != set_genes[name]:
        #     print(f"Warning: {name} length is {cds.seq_length()} instead of {set_genes[name]}.")
        # is_multiple = set_genes[name] % 3 == 0
        # print(f"Gene {name} is multiple of 3: {is_multiple}.")
    gtf_file.close()
    print(f"{count} genes with multiple hits, {count_not} found no match: {count_not / count * 100:.2f}%")
    print(f"{count_total} genes in total, {count_not} found no match: {count_not / count_total * 100:.2f}%")
    print('GTF file loaded.')
    return output_dico_cds


def main():
    output_path = "results/mutational_opportunities.tsv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dico_opp = defaultdict(list)
    codon_neighbors = build_codon_neighbors()

    fasta_path = "data/GCF_000699145.1_ASM69914v1_genomic.fna"
    gff_path = "data/GCF_000699145.1_ASM69914v1_genomic.gff"
    gene_stats = "data/geneStatsExp.tsv"
    nuc_matrix = "data/nucmat/JC.tsv"

    mutation_matrix = open_nuc_matrix(nuc_matrix)
    df_gene_stats = pd.read_csv(gene_stats, sep="\t")
    set_genes = {row["gene"]: row for _, row in df_gene_stats.iterrows()}
    fasta_chr = open_fasta(fasta_path)
    dict_cds = build_dict_cds(gff_path, fasta_chr, set_genes)

    count, count_not = 0, 0
    for seq_name, cds in dict_cds.items():
        seq = cds.seq
        if seq[:3] != "ATG":
            count_not += 1
        count += 1
        mu_syn, mu_nonsyn, n_total, mu_total = 0.0, 0.0, 0.0, 0.0
        for c_site in range(len(seq) // 3):
            ref_codon = seq[c_site * 3:c_site * 3 + 3]
            ref_aa = codontable[ref_codon]
            if ref_aa == "X" or ref_aa == "-":
                continue

            n_total += 1

            for (syn, ref_nuc, alt_nuc, alt_codon, alt_aa) in codon_neighbors[ref_codon]:
                mutation_rate = mutation_matrix[f"q_{ref_nuc}_{alt_nuc}"]
                assert np.isfinite(mutation_rate)
                mu_total += mutation_rate

                if syn:
                    mu_syn += mutation_rate
                else:
                    mu_nonsyn += mutation_rate

        dico_opp["gene"].append(seq_name)
        dico_opp["seq_len"].append(len(seq))
        dico_opp["mu_syn"].append(mu_syn)
        dico_opp["mu_nonsyn"].append(mu_nonsyn)
        dico_opp["mu_total"].append(mu_total)
        dico_opp["ratio"].append(mu_syn / mu_nonsyn)
        dico_opp["n_total"].append(n_total)

    df = pd.DataFrame(dico_opp)
    df.to_csv(output_path, index=False, sep="\t")
    print(f"{count_not} genes not starting with ATG: {count_not / count * 100:.2f}%")


if __name__ == '__main__':
    main()
