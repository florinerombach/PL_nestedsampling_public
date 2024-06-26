import matplotlib as plt
import pandas as pd
from model import *
from nested_sampling import *

def plot_fit(savepath, samples, input_df, data, param_dict, mode, param_input_in, samples_reweighted, param_vary_list, param_vary_keys, nfits):
    
    for sample_name in samples:

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))
        trpl_min = []

        indexes = input_df[input_df['sample']==sample_name].index

        for i in indexes:

            if input_df.loc[i, 'measurement'] == 'trpl':

                time, counts, noise_mask, exc_density = data[i]
                min_index = np.argmin(counts[noise_mask])
                trpl_min.append((time[min_index], counts[min_index]))      
                ax1.scatter(time, counts, alpha=(0.25),label=str(sample_name + " (" + "{:.2f}".format(exc_density) + "  cm⁻3)"))
                
                if mode == 'single':
                    TRPL_initial_fit, _ = calc_TRPL(time, i, input_df, param_dict, param_input_in, exc_density, show_carrier_densities = False)
                    ax1.plot(time, TRPL_initial_fit, linewidth=3, linestyle='dashed', label = 'in. fit'+str(i))
                elif mode == 'dist':
                    param_fit_draw = np.array([np.random.normal(np.mean(samples_reweighted[:,i]), np.std(samples_reweighted[:,i]), nfits) for i in range(len(param_vary_list))])
                    for n in range(nfits):
                        param_out = priors_to_inputdict(param_fit_draw[:,n], param_vary_keys)
                        TRPL_fit, _ = calc_TRPL(time, i, input_df, param_dict, param_out, exc_density, show_carrier_densities = False)
                        ax1.plot(time, TRPL_fit, linewidth=3, alpha=(0.1))

            elif input_df.loc[i, 'measurement'] == 'plqe':

                generation_rates, plqe = data[i]
                ax2.scatter(generation_rates, plqe*100, label = f'{sample_name} measured', marker = 'o')
                
                if mode == 'single':
                    PLQE_initial_fit, n, p, _ = PLQE_function(generation_rates, i, input_df, param_dict, param_input_in, print_carrier_densities = False, print_fitting_info = False)
                    ax2.scatter(generation_rates, PLQE_initial_fit*100, color = 'r', label = f'{sample_name} calculated', marker = 'x')
                elif mode == 'dist':
                    for n in range(nfits):
                        param_out = priors_to_inputdict(param_fit_draw[:,n], param_vary_keys)
                        PLQE_fit, n, p, _ = PLQE_function(generation_rates, i, input_df, param_dict, param_out, print_carrier_densities = False, print_fitting_info = False)
                        ax2.plot(generation_rates, PLQE_fit*100, linewidth=3, alpha=(0.1))

        fig.suptitle(sample_name+' initial fits')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_title('TRPL')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('PL (norm.)')
        lower_lim = trpl_min[np.argmin(np.array(trpl_min)[:,1])]
        ax1.set_ylim(lower_lim[1]/10, 1.2)
        ax1.set_xlim(10, lower_lim[0]*2)

        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_xlabel('Photon generation rate (photons s-1 cm-3)')
        ax2.set_ylabel('PL (norm.)')
        ax2.set_title('PLQE')

        ax1.legend()
        ax2.legend()

        plt.savefig(savepath / f'{sample_name}_initial_fit.png')
        plt.show()
