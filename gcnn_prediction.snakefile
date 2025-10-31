configfile: "gcnn_config.yaml"
PATH = "PATH"

rule all:
    input:
        expand(config[PATH]["OUTPUT"] + "{db}/r{N}_k{K}_ep{EP}_pl{PL}_d{D}_l{L}_po{PO}_lr{LR}/summary.txt", N=config["NUM"], K=config["KFOLD"], EP=config["EPOCHS"], PL=config["POOL"], D=config["DROPOUT"], L=config["LEVELS"], PO=config["POL_ORD"], LR=config["L_RATE"], db=config["DB"]),

rule run_741_ma: 
    input:
        gene_exp = config[PATH]["INPUT"] + "gene_expression_train_test_{db}.tsv",
        adj_mat = config[PATH]["INPUT"] + "adjacency_matrix_{db}.tsv",
        labels = config[PATH]["INPUT"] + "labels_train_test.tsv"
    output:
        concordance = config[PATH]["OUTPUT"] + "{db}/r{N}_k{K}_ep{EP}_pl{PL}_d{D}_l{L}_po{PO}_lr{LR}/predicted_concordance.csv",
        relevance = config[PATH]["OUTPUT"] + "{db}/r{N}_k{K}_ep{EP}_pl{PL}_d{D}_l{L}_po{PO}_lr{LR}/relevances_rendered_class.csv",
        summary = config[PATH]["OUTPUT"] + "{db}/r{N}_k{K}_ep{EP}_pl{PL}_d{D}_l{L}_po{PO}_lr{LR}/summary.txt"
    params:
        out_dir = directory(config[PATH]["OUTPUT"] + "{db}/"),
        res_dir = directory(config[PATH]["OUTPUT"] + "{db}/r{N}_k{K}_ep{EP}_pl{PL}_d{D}_l{L}_po{PO}_lr{LR}/"),
        EF = config["EV_FREQ"]
    shell:
        "if [ ! -d {params.out_dir} ] ; then mkdir {params.out_dir} ; fi ;"
        "if [ ! -d {params.res_dir} ] ; then mkdir {params.res_dir} ; fi ;"
        "python gcnn_prediction.py --gx {input.gene_exp} --adm {input.adj_mat} "
        "--l {input.labels} --o {params.res_dir} --k {wildcards.K} --ep {wildcards.EP} --ef {params.EF} "
        "--pl {wildcards.PL} --d {wildcards.D} --lv {wildcards.L} --po {wildcards.PO} --rs {wildcards.N} --lr {wildcards.LR}"