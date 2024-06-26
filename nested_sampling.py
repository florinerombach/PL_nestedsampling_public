import numpy as np
import dynesty
from collections import defaultdict
from dynesty import plotting as dyplot
from model import *


def priors_to_inputdict(priors, param_vary_keys):
    params_input = defaultdict(dict)

    if len(priors) != len(param_vary_keys):
        print('something went wrong')

    for i in range(len(priors)):
        key, sample = param_vary_keys[i].split('-')
        params_input[sample][key] = priors[i]

    return params_input


def log_likelihood(priors, param_vary_keys, param_dict, input_df, data):

    params_input = priors_to_inputdict(priors, param_vary_keys)

    LL = np.empty(len(input_df.index))

    for i in input_df.index:
        
        if input_df.loc[i, 'measurement'] == 'trpl':

            time, TRPL_data , noise_mask, exc_density = data[i]
            TRPL_calc, _ = calc_TRPL(time, i, input_df, param_dict, params_input, exc_density, show_carrier_densities = False)
            TRPL_sigma = np.sqrt(np.var(TRPL_data))  # should maybe be a parameter?

            TRPL_residsq = ((TRPL_calc  - TRPL_data)**2 / TRPL_sigma**2) #[noise_mask]
            TRPL_loglike = - 0.5 * np.sum(TRPL_residsq + np.log(2 * np.pi * TRPL_sigma**2))
            LL[i] = TRPL_loglike

        elif input_df.loc[i, 'measurement'] == 'plqe':

            generation_rates, PLQE_data = data[i]
            PLQE_calc, _, _, _ = PLQE_function(generation_rates, i, input_df, param_dict, params_input, print_carrier_densities = False, print_fitting_info = False)
            PLQE_sigma = 3e-4 # PLQE produced by noise - not quite right

            PLQE_residsq = ((PLQE_calc  - PLQE_data)**2 / PLQE_sigma**2)
            PLQE_loglike = - 0.5 * np.sum(PLQE_residsq + np.log(2 * np.pi * PLQE_sigma**2))
            LL[i] = PLQE_loglike
    
    return np.sum(LL)


def post_run_info(results, nr_live_points, savepath, param_vary_list):
    
    ndim = len(param_vary_list)

    # Save parameters of sampling
    runparams_save = pd.DataFrame({'Nr live points': nr_live_points, 
                                'Nr accepted points':results['niter'],
                                'Nr function calls':results['ncall'],
                                'Sampling efficiency': results['eff']})

    runparams_save.to_csv(savepath / 'nested_sampling_runparams.csv')

    # Save sampling run points
    origin_sample = results['samples_it']
    samples_out = results['samples']
    iter_bound = results['bound_iter']
    scalefactor = results['scale']
    loglike_out = results['logl']
    logvolume = results['logvol']
    logweights_out = results['logwt']
    logevidence = results['logz']
    logevidence_error = results['logzerr']
    information = results['information']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    samples_reweighted = dynesty.utils.resample_equal(samples_out, weights)

    results_save = pd.DataFrame({'active bound': iter_bound, 'sample origin iter': origin_sample, 
                            'scale factor active': scalefactor,
                            'log-likelihood': loglike_out, 'log-prior-volume': logvolume,
                            'logweights': logweights_out, 'logevidence': logevidence, 
                            'logevidence_error': logevidence_error, 'information': information})

    for i, param in enumerate(param_vary_list):
        results_save[f'{param[0]}'] = samples_reweighted[:,i]   

    print(results_save)
    results_save.to_csv(savepath / 'nested_sampling_results.csv')

    fig1, axes = dyplot.runplot(results, logplot = True)  # summary (run) plot
    fig1.tight_layout()
    fig1.savefig(savepath / 'runplot.png')

    param_names =  [p[0] for p in param_vary_list]
    fig2, axes = dyplot.traceplot(results, labels=param_names,
                                fig=plt.subplots(ndim, 2, figsize=(16, 25)))
    fig2.tight_layout()
    fig2.savefig(savepath / 'traceplot.png')

    fig3, axes = dynesty.dyplot.cornerplot(results, show_titles=True, 
                                title_kwargs={'y': 1.04}, labels=param_names,
                                quantiles=None, max_n_ticks=3,
                                fig=plt.subplots(ndim, ndim, figsize=(35, 35)))
    fig3.savefig(savepath / 'cornerplot.png')

    return samples_reweighted
