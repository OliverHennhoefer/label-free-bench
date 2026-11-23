# Rahman Siddiquee et al. 2026: AUCp

**Source.** "AUCp: Pseudo-AUC for Inference Model Selection with Unlabeled Validation Data in Abnormality Detection", IEEE TMI 2026, arXiv 2606.08742. https://arxiv.org/abs/2606.08742

**Establishes.** Label training normals 0 and every unlabeled test sample 1, then pick the checkpoint with the highest AUC against these pseudo-labels. In brain MRI it raises Brainomaly's Alzheimer's average AUC from 0.6421 (FID selection) to 0.6550, and HealthyGAN's headache AUC from 0.7695 to 0.8088.

**Setting.** Medical imaging; normal-only training plus an unlabeled mixed test set.

**Relevance.** Not directly usable: our selector sees no unlabeled test rows, only fitting and selection normals. The pseudo-AUC of training normals against selection normals would measure overfitting, not anomaly detection, so the closest analogue is a generalization-gap signal.

**Checked.** 2026-11-23, alphaXiv page and PDF via web fetch. Query: "arxiv 2606.08742 AUCp pseudo-AUC anomaly detection model selection".
