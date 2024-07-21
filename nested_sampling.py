import numpy as np
from scipy import stats
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

# Transforms the uniform random variables `u ~ Unif[0., 1.)` 
# to the parameters of interest according to distributions defined above. 
def prior_transform(u, param_vary_list):

    priors = np.array(u)  # copy u

    for i, param in enumerate(param_vary_list):
        if param[4] == 'flat':
            priors[i] = u[i]*(param[3]-param[2]) + param[2]
        elif param[4] == 'normal':
            priors[i] = stats.norm.ppf(u[i], np.mean(param[2:4]), (param[3]-param[2])/(2*2))

    return priors

def log_likelihood(priors, param_vary_keys, param_dict, input_df, data):

    params_input = priors_to_inputdict(priors, param_vary_keys)

    LL = np.empty(len(input_df.index))

    for i in input_df.index:

        y_calc, y_data, sigma = [None, None, None]
        
        if input_df.loc[i, 'measurement'] == 'trpl':

            time, y_data, noise_mask, exc_density, sigma = data[i]
            logspaced_indexes =  gen_log_space(len(time), round(len(time)/50) )
            y_calc, _ = calc_TRPL(time, i, input_df, param_dict, params_input, exc_density)

            noise_mask = noise_mask[logspaced_indexes]
            y_calc = y_calc[noise_mask]
            y_data = y_data[logspaced_indexes][noise_mask]
            sigma = sigma[logspaced_indexes][noise_mask]

        elif input_df.loc[i, 'measurement'] == 'plqe':

            generation_rates, y_data, sigma = data[i]
            y_calc, _, _, _ = PLQE_function(generation_rates, i, input_df, param_dict, params_input)

        LL[i] = - 0.5 * np.sum( ((y_calc - y_data)**2 / sigma**2) + np.log(2 * np.pi * sigma**2))
    
    #print(LL)
        
    return np.sum(LL)


def post_run_info(results, nr_live_points, bound, sample, savepath, param_vary_list):
    
    ndim = len(param_vary_list)

    # Save parameters of sampling
    runparams_save = pd.DataFrame({'Nr live points': nr_live_points, 'Sampling method': sample, 'Bounding method': bound,
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
    weights = results.importance_weights()
    samples_reweighted = dynesty.utils.resample_equal(samples_out, weights)

    results_save = pd.DataFrame({'active bound': iter_bound, 'sample origin iter': origin_sample, 
                            'scale factor active': scalefactor,
                            'log-likelihood': loglike_out, 'log-prior-volume': logvolume,
                            'logweights': logweights_out, 'logevidence': logevidence, 
                            'logevidence_error': logevidence_error, 'information': information})

    for i, param in enumerate(param_vary_list):
        results_save[f'{param[0]}'] = samples_reweighted[:,i]   
    param_names =  [p[0] for p in param_vary_list]

    print(results_save)
    results_save.to_csv(savepath / 'nested_sampling_results.csv')

    #dynesty.utils.jitter_run(res, rstate=None, approx=False)
    #dynesty.utils.quantile(x, q, weights=None)

    mean, cov = dynesty.utils.mean_and_cov(samples_out, weights)
    print('significant covariance found in', (np.abs(cov) > 0.3).nonzero())

    std = np.std(samples_reweighted, axis = 0)
    params_out = pd.DataFrame({'Parameter': param_names, 'Mean': mean, '2 stdev': 2*std})
    print(params_out)
    params_out.to_csv(savepath / 'nested_sampling_params_results.csv')

    fig1, axes = dyplot.runplot(results, logplot = True)  # summary (run) plot
    fig1.tight_layout()
    fig1.savefig(savepath / 'runplot.png')

    fig2, axes = dyplot.traceplot(results, labels=param_names,
                                fig=plt.subplots(ndim, 2, figsize=(16, 25)),
                                quantiles = [0.025, 0.5, 0.975],
                                smooth = 50, # posteriors are histograms with 50 bins
                                connect = True, # shows particle paths
                                show_titles = True # show quantiles
                                )
    fig2.tight_layout()
    fig2.savefig(savepath / 'traceplot.png')

    fig3, axes = dyplot.cornerplot(results,
                                labels=param_names,
                                quantiles = [0.025, 0.5, 0.975],
                                smooth = 50, # posteriors are histograms with 50 bins
                                show_titles = True
                                )
    fig3.savefig(savepath / 'cornerplot.png')

    return samples_reweighted