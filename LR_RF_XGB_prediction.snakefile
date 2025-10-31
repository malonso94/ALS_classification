configfile: "LR_RF_XGB_config.yaml"
PATH = "PATH"

rule all:
    input:
        expand(config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/{classifier}_{cv}_{k}_{db}_summary.txt", classifier=config["CLASSIFIER"], cv=config["CV"], k=config["K"], num=config["NUM"], db=config["DBS"], mlo=config["MLO"])

rule LR_RF_XGB_prediction: 
    input:
        gene_exp = config[PATH]["INPUT"] + "gene_expression_train_test_{db}.tsv",
        labels = config[PATH]["LABELS"] + "labels_train_test.tsv",
        val = config[PATH]["INPUT"] + "gene_expression_validation_{db}.tsv",
        vlab = config[PATH]["LABELS"] + "labels_validation.tsv"
    output:
        scores = config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/{classifier}_{cv}_{k}_{db}_scores.tsv",
        summary = config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/{classifier}_{cv}_{k}_{db}_summary.txt",
        feat_importance = config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/{classifier}_{cv}_{k}_{db}_feature_importance.tsv",
        val_pred = config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/{classifier}_{cv}_{k}_{db}_val_prediction.tsv",
        val_sum = config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/{classifier}_{cv}_{k}_{db}_val_scores.tsv"
    params:
        out_dir = directory(config[PATH]["OUTPUT"] + "{db}/"),
        res_dir = directory(config[PATH]["OUTPUT"] + "{db}/r{num}_ML_pred/")
    shell:
        "if [ ! -d {params.out_dir} ] ; then mkdir {params.out_dir} ; fi ;"
        "if [ ! -d {params.res_dir} ] ; then mkdir {params.res_dir} ; fi ;"
        "python LR_RF_XGB_prediction.py --l {input.labels} --gex {input.gene_exp} --cl {wildcards.classifier} "
        "--cv {wildcards.cv} --k {wildcards.k} --db {wildcards.db} --o {params.res_dir} --val {input.val} --vlab {input.vlab}"