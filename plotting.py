import matplotlib as plt
import pandas as pd
from scipy.interpolate import splrep, BSpline
import cmocean

from model import *
from nested_sampling import *

def plot_fit(savepath, samples, input_df, data, param_dict, mode, param_input_in, samples_reweighted, param_vary_list, param_vary_keys, nfits):
    
    for sample_name in samples:

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))
        trpl_min = []

        indexes = input_df[input_df['sample']==sample_name].index

        for i in indexes:

            if mode == 'dist':
                param_fit_draw = np.array([np.random.normal(np.mean(samples_reweighted[:,i]), np.std(samples_reweighted[:,i]), nfits) for i in range(len(param_vary_list))])
                param_fit_mean = np.mean(samples_reweighted, axis = 0)
                param_out_mean = priors_to_inputdict(param_fit_mean, param_vary_keys)

            if input_df.loc[i, 'measurement'] == 'trpl':

                time, counts, noise_mask, exc_density, sigma_trpl = data[i]
                logspaced_indexes = gen_log_space(len(time), round(len(time)/50) )

                min_index = np.argmin(counts[noise_mask])
                trpl_min.append((time[min_index], counts[min_index]))      
                ax1.scatter(time, counts, alpha=(0.25), color = 'b')

                spline = splrep(time[noise_mask], counts[noise_mask], s = len(counts[noise_mask]), k = 3, w = (1/sigma_trpl[noise_mask]))
                spline_fit = BSpline(*spline)(time[noise_mask])

                ax1.plot(time[noise_mask], spline_fit, c = 'g')
                ax1.plot(time[noise_mask], spline_fit+2*sigma_trpl[noise_mask], linewidth=1, linestyle='dashed', c = 'g')
                ax1.plot(time[noise_mask], spline_fit-2*sigma_trpl[noise_mask], linewidth=1, linestyle='dashed', c = 'g')

                if mode == 'single':
                    TRPL_initial_fit, _ = calc_TRPL(time, i, input_df, param_dict, param_input_in, exc_density, show_carrier_densities = False)
                    ax1.plot(time[logspaced_indexes], TRPL_initial_fit, linewidth=3, label = 'in. fit'+str(i), c = 'r')

                elif mode == 'dist':
                    TRPL_fit, _ = calc_TRPL(time, i, input_df, param_dict, param_out_mean, exc_density, show_carrier_densities = False)
                    ax1.plot(time[logspaced_indexes], TRPL_fit, linewidth=3, c = 'r')
                    for n in range(nfits):
                        param_out = priors_to_inputdict(param_fit_draw[:,n], param_vary_keys)
                        TRPL_fit, _ = calc_TRPL(time, i, input_df, param_dict, param_out, exc_density, show_carrier_densities = False)
                        ax1.plot(time[logspaced_indexes], TRPL_fit, linewidth=3, alpha=(0.1), c = 'r')

            elif input_df.loc[i, 'measurement'] == 'plqe':

                generation_rates, plqe, sigma_plqe = data[i]
                ax2.errorbar(generation_rates, plqe*100, yerr=2*sigma_plqe*100, color = 'b', alpha = 0.5, label = f'{sample_name} measured', fmt="o")
                
                if mode == 'single':
                    PLQE_initial_fit, n, p, _ = PLQE_function(generation_rates, i, input_df, param_dict, param_input_in, print_carrier_densities = False, print_fitting_info = False)
                    ax2.scatter(generation_rates, PLQE_initial_fit*100, color = 'r', label = f'{sample_name} calculated', marker = 'x')

                elif mode == 'dist':
                    PLQE_fit, n, p, _ = PLQE_function(generation_rates, i, input_df, param_dict, param_out_mean, print_carrier_densities = False, print_fitting_info = False)
                    ax2.scatter(generation_rates, PLQE_fit*100, color = 'r', label = f'{sample_name} calculated', marker = 'x')

                    for n in range(nfits):
                        param_out = priors_to_inputdict(param_fit_draw[:,n], param_vary_keys)
                        PLQE_fit, n, p, _ = PLQE_function(generation_rates, i, input_df, param_dict, param_out, print_carrier_densities = False, print_fitting_info = False)
                        ax2.plot(generation_rates, PLQE_fit*100, linewidth=3, alpha=(0.1), c = 'r')
                        
        fig.suptitle(sample_name+' initial fits')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_title('TRPL')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('PL (norm.)')
        if len(trpl_min) != 0:
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


def plot_sim(savepath, samples, input_df, data, param_dict, param_input, param_name, param_value_list, param_title, labels):
    color = cmocean.cm.phase(np.linspace(0,1,len(param_value_list)+4))
    #color = plt.cm.jet(np.linspace(0,1,len(param_value_list)+1))
    #color = cmc.batlow(np.linspace(0,1,len(param_value_list)+2))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
    
    fig.suptitle(param_title)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('TRPL', fontsize = 10)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('PL (norm.)')
    ax1.set_ylim(1e-3, 1.2)
    ax1.set_xlim(1e-9, 1e-3)

    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Generation rate (cm$^{-3}$ s$^{-1}$)')
    ax2.set_ylabel('PLQE (%)')
    ax2.set_title('PLQE', fontsize = 10)

    initial_param_value = param_input[samples[0]][param_name]

    for j in range(len(param_value_list)):
        print(param_name, param_value_list[j])

        param_input[samples[0]][param_name] = param_value_list[j]
    
        for i in [0,1]:

            if input_df.loc[i, 'measurement'] == 'trpl':

                time = np.linspace(0, int(1e6), int(1e6))
                logspaced_indexes =  gen_log_space(len(time), round(len(time)/50) )

                exc_density = 1e15
                TRPL_initial_fit, _ = calc_TRPL(time, i, input_df, param_dict, param_input, exc_density, show_carrier_densities = False)
                ax1.plot(1e-9 + time[logspaced_indexes]*1e-9, TRPL_initial_fit, linewidth=3, color = color[j+2], label = labels[j])

            elif input_df.loc[i, 'measurement'] == 'plqe':
                
                generation_rates = np.logspace(18, 22, 20)
                PLQE_initial_fit, n, p, _ = PLQE_function(generation_rates, i, input_df, param_dict, param_input, print_carrier_densities = False, print_fitting_info = False)
                ax2.plot(generation_rates[PLQE_initial_fit<=1], PLQE_initial_fit[PLQE_initial_fit<=1]*100, linewidth=3, color = color[j+2], label = labels[j])

    ax1.legend(prop={'size': 9}, loc = 'lower left')
    #ax2.legend(prop={'size': 8}, loc = 'lower left')

    plt.tight_layout()
    plt.savefig(savepath / f'{param_name}_vary.png')
    plt.show()

    param_input[samples[0]][param_name] = initial_param_value