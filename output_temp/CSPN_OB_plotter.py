import os 
import matplotlib.pylab as plt
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import shutil
import pandas as pd
import csv
from scipy import interpolate               # Required to regrid filter and model data

from dust_extinction.parameter_averages import F99   # Adopted dereddening law
import astropy.units as u

from collections import OrderedDict
from matplotlib.lines import Line2D

# Directories where relevant files are located
anc_directory = "anc"
sed_data_directory = "sed"


def make_CSPN_plots():
    ext = F99(Rv=3.1)
    catalogs = []
    bright_colours = sns.color_palette('bright')
    bright_colours.pop(7)
    markers = ['o', '+', '^', 'v', 'd', '1', '2', '3']

    comp_model_path = '0040000_6.00_HHe_0.9_0.1.dat.txt'
    comp_model_table = pd.read_csv(comp_model_path,dtype=float, sep=r'\s+', comment='#').values
    comp_model_table[:,1]*=comp_model_table[:,0]
    comp_model_table[:,1]*=comp_model_table[:,0]
    comp_model_table[:,1]/=2.99792458e21

    unred_fluxes = []
    comp_wavels = []
    for element in comp_model_table:
        if ((element[0] > 1000. ) and (element[0] < 30000.)):
            comp_wavels.append(element[0])
            unred_fluxes.append(element[1])

    # Loop through SED data files in the specified directory
    for file in os.listdir(os.fsencode(sed_data_directory)):
        filename = os.fsdecode(file)
        source=filename.replace('.sed', '')
        print(source)

        sed = np.genfromtxt(sed_data_directory + '/' + filename, delimiter="\t", names=True, dtype=None, encoding="utf-8")
        sed = np.sort(sed, order='wavel')
        sed = sed[sed['wavel']<30000]
        for i in range(len(sed)):
            if (sed[i]['catname'] not in catalogs):
                catalogs.append(sed[i]['catname'])

        ancillary=np.genfromtxt(anc_directory + "/" + filename.replace('sed', 'anc'), delimiter="\t", names=True, dtype=None, encoding="utf-8")
        adopted_ebv = ancillary[ancillary['parameter']=='E(B-V)']['value'][0]

        model_ebv_grid = pd.read_csv('TMAP_filter_convs_extinction_grid.csv', index_col=0)
        ebv_list = np.asarray(model_ebv_grid.index.tolist())
        
        uv_points = sed[(sed['wavel']<3000.) & (sed['unsub_flux']>0.)]
        uv_filter_names = uv_points['svoname']
        ebv_grid_selection = model_ebv_grid[uv_filter_names].to_numpy()
        ebv_grid_selection = np.log10(ebv_grid_selection)
        grid_interp = interpolate.interp1d(ebv_list, ebv_grid_selection, axis=0)
        
        reddened_tmap_uv_fluxes = grid_interp(adopted_ebv)
        reddened_tmap_uv_fluxes += 26. #Convert to Janskys
        observed_uv_fluxes = uv_points['unsub_flux']
        observed_uv_fluxes = np.log10(observed_uv_fluxes)
        offset = np.median(observed_uv_fluxes - reddened_tmap_uv_fluxes)
        reddened_tmap_uv_fluxes += offset
        uv_wavels = sed[(sed['wavel']<3000.) & (sed['unsub_flux']>0.)]['wavel']

        ext = F99(Rv=3.1)
        red_spectra = np.log10(unred_fluxes * ext.extinguish(comp_wavels*u.AA, Ebv=adopted_ebv))

        interp=interpolate.interp1d(comp_wavels, red_spectra)
        ext_model_alignment_fluxes=interp(uv_wavels)
        offset = np.median(reddened_tmap_uv_fluxes-ext_model_alignment_fluxes)
        red_spectra += offset

        OB_fitted_points = sed[sed['mask']==True]
  
        fig, (topfig, bottomfig) = plt.subplots(2)
        fig.set_size_inches(4.5, 5.5)
        topfig.plot(comp_wavels, red_spectra, color='red', linewidth=0.4, label='TMAP Spectrum', zorder=-1)
        topfig.scatter(uv_wavels, reddened_tmap_uv_fluxes, marker='s', color='red', label='TMAP Filter Convolutions', zorder=-1)
        bottomfig.plot(OB_fitted_points['wavel'], np.log10(OB_fitted_points['model']), linewidth=0.4, marker='s', color='blue', label='BT-SETTL Filter Convolutions', zorder=-2)
        i = 0
        for catalog in catalogs:
            unmasked_points = False
            cat_points = sed[sed['catname']==catalog]
            cat_true_points = cat_points[cat_points['mask']==True]
            cat_false_points = cat_points[cat_points['mask']==False]
            if (len(cat_true_points) > 0):
                bottomfig.scatter(cat_true_points[cat_true_points['wavel']>3000.]['wavel'], np.log10(cat_true_points[cat_true_points['wavel']>3000.]['dered']), marker=markers[i%8], color=bright_colours[i%9],label=catalog, zorder=1)
                topfig.scatter(cat_true_points['wavel'], np.log10(cat_true_points['unsub_flux']), marker=markers[i%8], label=catalog, color=bright_colours[i%9], zorder=0)
                unmasked_points = True
            if (unmasked_points and (len(cat_false_points) > 0)):
                topfig.scatter(cat_false_points['wavel'], np.log10(cat_false_points['unsub_flux']), marker=markers[i%8], color=bright_colours[i%9], label='_nolegend_', zorder=0)
                bottomfig.scatter(cat_false_points[cat_false_points['wavel']>3000.]['wavel'], np.log10(cat_false_points[cat_false_points['wavel']>3000.]['dered']), marker=markers[i%8], color='grey', label='_nolegend_', zorder=0)
            elif not unmasked_points and (len(cat_false_points) > 0):
                topfig.scatter(cat_false_points['wavel'], np.log10(cat_false_points['unsub_flux']), marker=markers[i%8], color=bright_colours[i%9], label=catalog, zorder=0)
                bottomfig.scatter(cat_false_points[cat_false_points['wavel']>3000.]['wavel'], np.log10(cat_false_points[cat_false_points['wavel']>3000.]['dered']), marker=markers[i%8], color='grey', zorder=0)
            i+=1
        topfig.set_xscale('log')
        bottomfig.set_xscale('log')
        bottomfig.set_xlabel(r'Wavelength [$\AA$]')
        topfig.set_ylabel('Observed log(F)')
        bottomfig.set_ylabel('Subtracted and dereddened log(F)')
        min_unsub_flux = min(i for i in sed['unsub_flux'] if i > 0)
        topfig.set_ylim(np.log10(min_unsub_flux)-0.5, np.log10(max(sed['unsub_flux']))+0.3)
        bottomfig.set_ylim(np.log10(min(sed[sed['wavel']>3000.]['dered']))-0.25, np.log10(max(sed[sed['wavel']>3000.]['dered']))+0.25)
        bottomfig.set_xlim(topfig.get_xlim())
        bottomfig.text(.99, .01, r'G-Tomo E(B-V): '+"{:.2E}".format(adopted_ebv)+'\n'
                   +source.replace('_', ' '), ha='right', va='bottom', fontsize=10, transform=bottomfig.transAxes)
        # Collect legend handles and labels from both axes
        handles_top, labels_top = topfig.get_legend_handles_labels()
        handles_bottom, labels_bottom = bottomfig.get_legend_handles_labels()

        # Combine and deduplicate
        all_handles = handles_top + handles_bottom
        all_labels = labels_top + labels_bottom

        # Use dict to remove duplicates while preserving order
        from collections import OrderedDict
        unique = list(OrderedDict(zip(all_labels, all_handles)).items())
        
        item = unique.pop(-1)  # Remove last item
        unique.insert(2, item)  # Insert it at index 2
        unique_labels, unique_handles = zip(*unique)

        # Add a figure-level legend below the plots
        fig.legend(unique_handles, unique_labels,
                loc='lower center', bbox_to_anchor=(0.5, 0.),
                ncol=2, fontsize='small', frameon=False)
        # Adjust layout to fit the legend
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.285)
        fig.savefig('OB_plots/'+filename.replace('sed', 'pdf'))
        plt.close()
    return


if (__name__ == "__main__"):
    make_CSPN_plots()