Multichannel nonnegative matrix factorization (NMF) toolbox
-----------------------------------------------------------

------------------------------------------------------------------------------------------------------
Copyright 2009 Alexey Ozerov and Cedric Fevotte
(alexey.ozerov -at- telecom-paristech.fr, cedric.fevotte -at- telecom-paristech.fr)

This software is distributed under the terms of the GNU Public License
version 3 (http://www.gnu.org/licenses/gpl.txt)

This software contains implementations of algorithms described in the paper, i.e.,
    - Expectation Maximization (EM) algorithm (for both instantaneous and onvolutive cases)
    - Multiplicative Update (MU) rules (for both instantaneous and onvolutive cases)

A. Ozerov and C. Fevotte,
"Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf
------------------------------------------------------------------------------------------------------
Some Matlab routines used are from
Signal Separation Evaluation Campaign (SiSEC 2008) - "Under-determined speech and music mixtures" task:
    http://sisec.wiki.irisa.fr/tiki-index.php?page=Under-determined+speech+and+music+mixtures
------------------------------------------------------------------------------------------------------



INSTALLATION
------------

In order to be able running example_ssep_SiSEC08_conv
one needs to download the following files:
    - estmix_conv_5cm.p
    - estmix_conv_5cm.readme
from SiSEC 2008 (Under-determined speech and music mixtures) web page:
    - http://sisec.wiki.irisa.fr/tiki-index.php?page=Under-determined+speech+and+music+mixtures
and to put them with other Matlab files (e.g., into aux_tools directory).



MAIN FUNCTIONS
--------------

multinmf_inst_em.m - EM algorithm for multichannel NMF (linear instantaneous mixtures)
multinmf_conv_em.m - EM algorithm for multichannel NMF (convolutive mixtures)
multinmf_inst_mu.m - MU rules for multichannel NMF (linear instantaneous mixtures)
multinmf_conv_mu.m - MU rules for multichannel NMF (convolutive mixtures)



EXAMLES OF USAGE AND CODE USED FOR OUR SiSEC 2008 SUBMISSION
------------------------------------------------------------

example_usage_multinmf_inst_em.m - example of usage of multinmf_inst_em.m
example_usage_multinmf_conv_em.m - example of usage of multinmf_conv_em.m
example_usage_multinmf_inst_mu.m - example of usage of multinmf_inst_mu.m
example_usage_multinmf_conv_mu.m - example of usage of multinmf_conv_mu.m

example_ssep_SiSEC08_inst.m - a variant of code used for SiSEC 2008 submission
                              for "Under-determined speech and music mixtures / instantaneous mixtures" task
                              by A. Ozerov and C. Févotte (Algorithm 2):
                              http://www.irisa.fr/metiss/SiSEC08/SiSEC_underdetermined/results/aozerov/contact.txt
example_ssep_SiSEC08_conv.m - a similar code for convolutive mixtures (was not submitted to SiSEC 2008)



OTHER FUNCTIONS
---------------


aux_tools/Init_KL_NMF_fr_sep_sources.m - initialize NMF from STFTs of separated sources using KL divergence
aux_tools/multinmf_recons_im.m         - reconstructs source images from multi-nmf factorization (with MU rules)
aux_tools/nmf_kl.m                     - NMF decomposition using MU rules and KL divergence



FUNCTIONS FROM SiSEC 2008
-------------------------

The following Matlab routines are from
Signal Separation Evaluation Campaign (SiSEC 2008) - "Under-determined speech and music mixtures" task:
    http://sisec.wiki.irisa.fr/tiki-index.php?page=Under-determined+speech+and+music+mixtures

aux_tools/bss_eval_images.m - BSS performances of estimated source spatial image signals
aux_tools/stft_multi.m      - multichannel STFT
aux_tools/estmix_inst.m     - mixing matrix estimation for instantaneous mixtures
aux_tools/sep_binmask.m     - source STFT estimation via binary masking
aux_tools/sep_l0min.m       - source STFT estimation via lp-norm minimization
aux_tools/istft_multi.m     - multichannel inverse STFT
aux_tools/src_image.m       - computation of the source spatial images in the STFT domain
