# Ma, Zhao, Zhang, Akoglu 2021: do internal strategies suffice?

**Source.** "A Large-scale Study on Unsupervised Outlier Model Selection: Do Internal Strategies Suffice?" arXiv preprint, v2 April 2021. https://arxiv.org/abs/2104.01422

**Establishes.** Seven label-free internal strategies (cluster indices XB/RS/CH, EM/MV, IREOS, and the consensus methods UDR, model centrality, HITS, ENS) were compared on a pool of 297 PyOD models over 39 tabular ODDS/DAMI datasets. No stand-alone strategy is significantly better than random selection. Consensus methods beat random but do not differ significantly from Isolation Forest with random hyperparameters.

**Setting.** Unsupervised, contaminated fitting data; labels only for evaluation. Not normal-only, and a much larger pool than ours.

**Relevance.** The prior negative baseline. Our library baselines implement most of these strategies; expect them to land near `uniform_model` or default Isolation Forest. A new mechanism should be judged against a strong fixed control, not only against random.

**Checked.** 2026-11-23, abstract and PDF via web fetch. Query: "arxiv 2104.01422 Ma Zhao Akoglu model selection outlier detection internal strategies".
