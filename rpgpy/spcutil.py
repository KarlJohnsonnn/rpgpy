import numpy as np
from numba import jit

def spectra2moments(LV0data, LV0meta, **kwargs):
    """
    This routine calculates the radar moments: reflectivity, mean Doppler velocity, spectrum width, skewness and
    kurtosis from the level 0 spectrum files of the 94 GHz RPG cloud radar.

    Args:
        LV0data (dict): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        LV0meta (dict): information from params_[campaign].toml for the system LIMRAD94

    Return:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """
    from time import time
    # initialize variables:
    n_ts, n_rg = LV0data['TotSpec'].shape[:2]
    Z = np.full((n_ts, n_rg), np.nan)
    V = np.full((n_ts, n_rg), np.nan)
    SW = np.full((n_ts, n_rg), np.nan)
    SK = np.full((n_ts, n_rg), np.nan)
    K = np.full((n_ts, n_rg), np.nan)

    spec_lin = LV0data['TotSpec'].copy()
    mask = spec_lin <= 0.0
    spec_lin[mask] = 0.0

    # combine the mask for "contains signal" with "signal has more than 1 spectral line"
    mask2D = np.all(mask, axis=2)
    ranges = np.append(LV0meta['RngOffs'], LV0meta['RAltN'])

    for iC in range(LV0meta['SequN']):
        Dopp_res = np.mean(np.diff(LV0meta[f'C{iC+1}vel']))
        if iC == 2:
            Dopp_res = 1.0
        tstart = time()
        for iR in range(ranges[iC], ranges[iC + 1]):  # range dimension
            for iT in range(n_ts):  # time dimension
                if mask2D[iT, iR]: continue
                _, (lb, rb) = find_peak_edges(spec_lin[iT, iR, :])
                Z[iT, iR], V[iT, iR], SW[iT, iR], SK[iT, iR], K[iT, iR] = radar_moment_calculation(
                    spec_lin[iT, iR, lb:rb], LV0meta[f'C{iC+1}vel'][lb:rb]
                )
                V[iT, iR] -= Dopp_res / 2.0  # values at center of each bin

        print(f'Chirp {iC + 1} Moments Calculated, elapsed time = {time() - tstart} [min:sec]')

    SW[:, ranges[2]:] *= 2

    moments = {'Ze': Z, 'MeanVel': V, 'SpecWidth': SW, 'Skewn': SK, 'Kurt': K}
    # create the mask where invalid values have been encountered
    invalid_mask = np.full((LV0data['TotSpec'].shape[:2]), True)
    invalid_mask[np.where(Z > 0.0)] = False

#    # despeckle the moments
#    if 'despeckle' in kwargs and kwargs['despeckle']:
#        tstart = time.time()
#        # copy and convert from bool to 0 and 1, remove a pixel  if more than 20 neighbours are invalid (5x5 grid)
#        new_mask = despeckle(invalid_mask, 80.)
#        invalid_mask[new_mask] = True
#        logger.info(f'Despeckle done, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    # mask invalid values with fill_value = -999.0
    for mom in moments.keys():
        moments[mom][invalid_mask] = -999.0


    return moments


@jit(nopython=True, fastmath=True)
def radar_moment_calculation(signal, vel_bins):
    """
    Calculation of radar moments: reflectivity, mean Doppler velocity, spectral width,
        skewness, and kurtosis of one Doppler spectrum. Optimized for the use of Numba.

    Note:
        Divide the signal_sum by 2 because vertical and horizontal channel are added.
        Subtract half of of the Doppler resolution from mean Doppler velocity, because

    Args:
        - signal (float array): detected signal from a Doppler spectrum
        - vel_bins (float array): extracted velocity bins of the signal (same length as signal)

    Returns:
        dict containing

            - Ze_lin (float array): reflectivity (0.Mom) over range of velocity bins [mm6/m3]
            - VEL (float array): mean velocity (1.Mom) over range of velocity bins [m/s]
            - sw (float array):: spectrum width (2.Mom) over range of velocity bins [m/s]
            - skew (float array):: skewness (3.Mom) over range of velocity bins
            - kurt (float array):: kurtosis (4.Mom) over range of velocity bins
    """

    signal_sum = np.sum(signal)  # linear full spectrum Ze [mm^6/m^3], scalar
    Ze_lin = signal_sum / 2.0
    pwr_nrm = signal / signal_sum  # determine normalized power (NOT normalized by Vdop bins)

    VEL = np.sum(vel_bins * pwr_nrm)
    vel_diff = vel_bins - VEL
    vel_diff2 = vel_diff * vel_diff
    sw = np.sqrt(np.abs(np.sum(pwr_nrm * vel_diff2)))
    sw2 = sw * sw
    skew = np.sum(pwr_nrm * vel_diff * vel_diff2 / (sw * sw2))
    kurt = np.sum(pwr_nrm * vel_diff2 * vel_diff2 / (sw2 * sw2))

    return Ze_lin, VEL, sw, skew, kurt


@jit(nopython=True, fastmath=True)
def find_peak_edges(signal, threshold=-1, imaxima=-1):
    """Returns the indices of left and right edge of the main signal peak in a Doppler spectra.

    Args:
        signal (numpy.array): 1D array Doppler spectra
        threshold: noise threshold

    Returns (list):
        [index_left, index_right]: indices of signal minimum/maximum velocity
    """
    len_sig = len(signal)
    index_left, index_right = 0, len_sig
    if threshold < 0: threshold = np.min(signal)
    if imaxima < 0: imaxima = np.argmax(signal)

    for ispec in range(imaxima, len_sig):
        if signal[ispec] > threshold: continue
        index_right = ispec
        break

    for ispec in range(imaxima, -1, -1):
        if signal[ispec] > threshold: continue
        index_left = ispec + 1  # the +1 is important, otherwise a fill_value will corrupt the numba code
        break

    return threshold, [index_left, index_right]
