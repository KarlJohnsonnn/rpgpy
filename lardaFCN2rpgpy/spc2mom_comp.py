import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# 20190801 Punta-Arenas
#LV0DATA_PATH = './data/Y2019/M08/D01/190801_070001_P05_ZEN.LV0'
#LV1DATA_PATH = './data/Y2019/M08/D01/190801_070001_P05_ZEN.LV1'


# 20200523 Leipzig
#LV0DATA_PATH = './data/Y2020/M05/D23/200523_120001_P05_ZEN.LV0'
#LV1DATA_PATH = './data/Y2020/M05/D23/200523_120001_P05_ZEN.LV1'


# 20201220 Leipzig
LV0DATA_PATH = './data/Y2020/M12/D20/201220_170000_P05_ZEN.LV0'
LV1DATA_PATH = './data/Y2020/M12/D20/201220_170000_P05_ZEN.LV1'


LARDA_PATH = '/Users/willi/code/python/larda3/larda/'
RPGPY_PATH = '/Users/willi/code/python/local_stuff/rpgpy/'
QUICKLOOKS = join(RPGPY_PATH, 'lardaFCN2rpgpy/quicklooks/')
fig_name = join(QUICKLOOKS, f'Comparison_LIMRAD94.png')
sys.path.append(LARDA_PATH)
sys.path.append(RPGPY_PATH)

from datetime import datetime
from rpgpy import read_rpg
from rpgpy.spcutil import spectra2moments
import pyLARDA.Transformations as tr

MOMENT_LIST = ['Ze', 'MeanVel', 'SpecWidth', 'Skewn', 'Kurt']


def rpgpy_to_xarray(header, data):

    ds = xr.Dataset(
        {mom: (["ts", "rg"], data[mom]) for mom in MOMENT_LIST},
        coords={
            "ts": data['Time'] + (datetime(2001, 1, 1) - datetime(1970, 1, 1)).total_seconds(),
            "rg": header['RAlts']
        },
        attrs={
            'rg_unit': 'm',
            'var_unit': '',
            'system': '',
            'var_lims': [-1, 1],
            'colormap': 'jet',
            'Fill_Value': -999.0,
        }
    )

    mask = ds['Ze'] <= 0.0

    for mom in MOMENT_LIST:
        ds[mom].attrs = ds.attrs.copy()
        ds[mom].values[mask] = -999.0

    ds['Ze'].attrs['var_lims'], ds['Ze'].attrs['var_unit']  = [-50, 20], 'dBZ'
    ds['MeanVel'].attrs['var_lims'], ds['MeanVel'].attrs['var_unit'] = [-4, 2], 'ms-1'
    ds['SpecWidth'].attrs['var_lims'], ds['SpecWidth'].attrs['var_unit'] = [0, 1], 'ms-1'
    ds['Skewn'].attrs['var_lims'], ds['Skewn'].attrs['var_unit'] = [-1, 1], '-'
    ds['Kurt'].attrs['var_lims'], ds['Kurt'].attrs['var_unit'] = [0, 4], '-'

    return ds


header0, data0 = read_rpg(LV0DATA_PATH)
header1, data1 = read_rpg(LV1DATA_PATH)

moments = spectra2moments(data0, header0)
moments.update({'Time': data0['Time']})

ds0 = rpgpy_to_xarray(header0, moments)
ds1 = rpgpy_to_xarray(header1, data1)


def fcndiff(a, b):
    c = a.copy()
    mask = (a.values > -90.0) * (b.values > -90.0)
    c.values = np.full(a.shape, -999.)
    c.values[mask] = a.values[mask] - b.values[mask]
    return c


def compare_RPG(ds0, ds1):
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr


    MOMENT_LIST = ['Ze', 'MeanVel', 'SpecWidth', 'Skewn', 'Kurt']
    nr, nc = 5, 3
    fig, ax = plt.subplots(nrows=nr, ncols=nc)
    kwargs = {
        'range_interval': [0, 12],
        'fontweight': 'normal',
        'fontsize': 6,
        'labelsize': 6,
        'rg_converter': True,
        'zlabel': '',
        'figsize': [20, 15]
    }

    tol = 1.0e-2
    tols = [-tol, tol]
    mask = ds0['Ze'] > 0.0

    fig, ax[0, 0] = tr.plot_timeheight2(ds0['Ze'], fig=fig, ax=ax[0, 0], var_converter='lin2z', cbar=False, **kwargs)
    fig, ax[0, 1] = tr.plot_timeheight2(ds1['Ze'], fig=fig, ax=ax[0, 1], var_converter='lin2z', cbar=False, **kwargs)
    dsX = fcndiff(ds0['Ze'], ds1['Ze'])
    dsX.attrs['colormap'] = 'coolwarm'
    fig, ax[0, 2] = tr.plot_timeheight2(dsX, fig=fig, ax=ax[0, 2], var_lims=tols,cmap='coolwarm',  **kwargs)

    for i in range(nr):
        kwargs.update({'cbar': False if nc < 2 else True})
        if i > 0:
            mom = MOMENT_LIST[i]
            fig, ax[i, 0] = tr.plot_timeheight2(ds0[mom], fig=fig, ax=ax[i, 0], **kwargs)
            fig, ax[i, 1] = tr.plot_timeheight2(ds1[mom], fig=fig, ax=ax[i, 1], **kwargs)
            dsX = fcndiff(ds0[mom], ds1[mom])
            dsX.attrs['colormap'] = 'coolwarm'
            fig, ax[i, 2] = tr.plot_timeheight2(dsX, fig=fig, ax=ax[i, 2], var_lims=tols, **kwargs)

        if i < nr-1:
            ax[i, 0].set(xlabel='', xticklabels=[])
            ax[i, 1].set(xlabel='', xticklabels=[])
            ax[i, 2].set(xlabel='', xticklabels=[])
        ax[i, 1].set_ylabel('')
        if nc > 2: ax[i, 2].set_ylabel('')

    ax[0, 0].set_title('calculated moments from spectra lv0 binary', fontsize=kwargs['fontsize'])
    ax[0, 1].set_title('moments from lv1 binary', fontsize=kwargs['fontsize'])
    ax[0, 2].set_title('difference moments from (lv0 - lv1) binary', fontsize=kwargs['fontsize'])
    return fig, ax

fig, ax = compare_RPG(ds0, ds1)
fig.savefig(fig_name, dpi=450)
print(f"   Saved to  {fig_name}")


ranges = np.append(header0['RngOffs'], header0['RAltN'])
for mom in MOMENT_LIST:
    for iC in range(header0['SequN']):
        Dopp_res = np.mean(np.diff(header0[f'C{iC + 1}vel']))
        dsX = fcndiff(ds0[mom], ds1[mom])
        dsX.attrs['colormap'] = 'coolwarm'
        print(Dopp_res, np.sqrt(Dopp_res))
        print(f'meandiff({mom})({iC}) = {np.mean(dsX.values[:, ranges[iC]:ranges[iC + 1]])}\t')