# Nested Sampling for Bayesian Inference and Model Comparison on PL Data

This notebook demonstrates inference on photoluminescence data of thin film semiconductors using a custom physical model. The nested sampling algorithm used provides both a posterior distribution for the physical parameters used in the model, and the total evidence obtained by the sampling (to be used for model comparison). 

The nested sampling package used is dynesty, developed by Josh Speagle and contributors. A list of paper citations for the particular version of the sampler used can be found at the end of the notebook. I have also drawn upon various references for model development, which are provided in the model.py section where relevant.

© Copyright 2024, Florine Rombach (University of Oxford). Data belongs to the University of Oxford.