configfile: "gcnn_validation_config.yaml"
PATH = "PATH"
wildcard_constraints:
    n="\d"

rule all:
    input:
        expand(config[PATH]["OUTPUT"] + "{db}/r{n}_{pms}/pred_validation.tsv", n=config["NUM"], pms=config["PARAMS"], db=config["DB"])

rule run_validation: 
    input:
        gene_exp = config[PATH]["INPUT"] + "gene_expression_validation_{db}.tsv",
        ady_mat = config[PATH]["INPUT"] + "adjacency_matrix_{db}.tsv",
        labels = config[PATH]["INPUT"] + "labels_validation.tsv"
    output:
        config[PATH]["OUTPUT"] + "{db}/r{n}_{pms}/pred_validation.tsv"
    params:
        out_dir = config[PATH]["OUTPUT"] + "{db}/r{n}_{pms}/",
    shell:
        "python gcnn_validation.py --gx {input.gene_exp} --adm {input.ady_mat} --l {input.labels} "
        "--p {wildcards.pms} --o {params.out_dir} --n {wildcards.n} --db {wildcards.db}" 