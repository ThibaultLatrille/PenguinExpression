wget https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/9233/101/GCF_000699145.1_ASM69914v1/GCF_000699145.1_ASM69914v1_cds_from_genomic.fna.gz -O data/GCF_000699145.1_ASM69914v1_cds_from_genomic.fna.gz
python3 mutational_opportunities.py
python3 theta.py
python3 merge.py
python3 analysis.py