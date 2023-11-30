import numpy as np
from collections import defaultdict
import pandas as pd

sfs_weight = {"watterson": lambda i, n: 1.0 / i, "tajima": lambda i, n: n - i, "fay_wu": lambda i, n: i}


def theta(sfs_epsilon, daf_n, weight_method):
    assert len(sfs_epsilon) == daf_n - 1
    sfs_theta = sfs_epsilon * np.array(range(1, daf_n))
    weights = np.array([sfs_weight[weight_method](i, daf_n) for i in range(1, daf_n)])
    return np.sum(sfs_theta * weights) / np.sum(weights)


def intergenetic(intergenic_path, intergenic_output_path):
    intergenic_df = pd.read_csv(intergenic_path, sep='\t')
    intergenic_outdict = defaultdict(list)
    for pop in ["e", "k"]:
        for region in ["Intergenic", "Introns"]:
            sfs = intergenic_df[f"{pop}{region}"]
            intergenic_outdict["pop"].append(pop)
            intergenic_outdict["region"].append(region)
            for theta_method in sfs_weight:
                theta_sfs = theta(sfs, 48, theta_method)
                intergenic_outdict[theta_method].append(theta_sfs)
    intergenic_outdf = pd.DataFrame(intergenic_outdict)
    print(f"Saving {intergenic_output_path}")
    intergenic_outdf.to_csv(intergenic_output_path, sep='\t', index=False)


def main(input_path, output_path):
    input_df = pd.read_csv(input_path, sep='\t', names=["pop", "gene", "len", "type", "DAC"])
    max_dac = 48
    output_dict = defaultdict(list)
    for _, row in input_df.iterrows():
        pop, gene, type_mut, dac = row["pop"], row["gene"], row["type"], row["DAC"]
        dac_strip = row["DAC"].replace("[", "").replace("]", "")
        if len(dac_strip) == 0:
            dac = []
        else:
            dac = [int(x) for x in dac_strip.split(",")]
        sfs = [0] * max_dac
        for snp_dac in dac:
            assert snp_dac < max_dac
            sfs[snp_dac] += 1
        sfs = sfs[1:]
        output_dict["gene"].append(gene)
        output_dict["pop"].append(pop)
        output_dict["type"].append(type_mut)
        for theta_method in sfs_weight:
            theta_sfs = theta(sfs, max_dac, theta_method)
            output_dict[theta_method].append(theta_sfs)
    output_df = pd.DataFrame(output_dict)
    print(f"Saving {output_path}")
    output_df.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    intergenetic("data/sfs_intergenic_introns.tsv", "results/theta_introns.tsv")
    intergenetic("data/sfs_intergenic_introns_allVariants.tsv", "results/theta_intergenic_allVariants.tsv")
    main(input_path="data/dacXgene.tsv", output_path="results/theta.tsv")
