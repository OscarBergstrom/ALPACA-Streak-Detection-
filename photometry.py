"""Streak section profile fits, background strips, errors, outlier masks, light curve."""
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from photutils.aperture import RectangularAperture, aperture_photometry
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

# Method bodies are copied unchanged from the original single class.
# They still use self, so they only work as part of OperatingFitsFiles (frame.py).


class PhotometryMixin:
    def _bilinear_sample(self, data, xs, ys):
        """Sample `data` at fractional pixel coordinates (xs, ys) via bilinear interpolation."""
        # map_coordinates expects [row, col] order, i.e. [y, x]
        coords = np.array([ys, xs])
        return map_coordinates(data, coords, order=1, mode='constant', cval=np.nan)

    def fit_section_profile(self, c, t, beta_2d, half_width=15, n_samples=7, sample_spacing=1.0):
        """
        Perpendicular Moffat fit for one section, centred at c with tangent angle t.
        Co-adds several parallel cross-cuts sampled along the tangent direction
        (median-stacked, with per-offset scatter used as fit weights) to improve
        SNR while staying local enough to avoid smearing across streak curvature.
    
        beta is held fixed (not fit) — the Moffat wing-shape parameter is poorly
        constrained by a single local section even at good S/N, since it's set by
        the low-flux wings specifically. Determine beta once from a Moffat fit to
        bright, isolated calibration stars and pass it in here, rather than
        re-fitting it per section.
    
        Returns corrected centre, equivalent Gaussian sigma (derived from the
        fitted Moffat FWHM, so downstream code that expects a sigma is unaffected),
        amplitude, mu offset, fit-success flag, and mu uncertainty.
        """
        p_lsf = beta_2d - 0.5
        if p_lsf <= 0.5:
            raise ValueError(f"beta_2d={beta_2d:.2f} gives a non-normalisable LSF.")
        px, py = -np.sin(t), np.cos(t)   # perpendicular direction
        tx, ty = np.cos(t), np.sin(t)    # tangent direction
    
        offsets = np.arange(-half_width, half_width + 1)
    
        # sample multiple parallel cuts along the tangent, centred on c
        along_offsets = (np.arange(n_samples) - n_samples // 2) * sample_spacing
        profiles = []
        for a in along_offsets:
            cx, cy = c[0] + a * tx, c[1] + a * ty
            xs = cx + offsets * px
            ys = cy + offsets * py
            profiles.append(self._bilinear_sample(self.fits_data, xs, ys))
        profiles = np.array(profiles)
    
        profile = np.nanmedian(profiles, axis=0)
        profile_scatter = np.nanstd(profiles, axis=0)
        # avoid zero-weight offsets (e.g. n_samples=1, or all cuts agreeing exactly)
        valid_scatter = profile_scatter[profile_scatter > 0]
        fallback_scatter = np.nanmedian(valid_scatter) if valid_scatter.size else 1.0
        profile_scatter[profile_scatter == 0] = fallback_scatter
    
        good = np.isfinite(profile) & np.isfinite(profile_scatter)
    
        A0 = np.nanmax(profile) - np.nanmedian(profile)
        B0 = np.nanmedian(profile)
        alpha0 = (half_width / 4) / np.sqrt(2 ** (1 / p_lsf) - 1)  # match Gaussian-equiv width
        p0 = [A0, 0.0, alpha0, B0]
    
        def _moffat(y, A, mu, alpha, B):
            return A * (1 + ((y - mu) / alpha) ** 2) ** (-p_lsf) + B
    
        try:
            if good.sum() < 4:  # not enough points to fit 4 params
                raise RuntimeError("too few valid samples")
    
            popt, pcov = curve_fit(
                _moffat,
                offsets[good],
                profile[good],
                p0=p0,
                sigma=profile_scatter[good],
                absolute_sigma=True,
                bounds=(
                    [-np.inf, -half_width, 0.1, -np.inf],
                    [np.inf, half_width, half_width, np.inf],
                ),
            )
            A_fit, mu_fit, alpha_fit, bkg_fit = popt
            alpha_fit = abs(alpha_fit)
    
            # convert Moffat alpha -> equivalent Gaussian sigma via matched FWHM
            fwhm = 2 * alpha_fit * np.sqrt(2 ** (1 / p_lsf) - 1)
            sigma_fit = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
            # propagate alpha uncertainty into the derived sigma
            alpha_err = np.sqrt(pcov[2, 2])
            sigma_err = alpha_err * (sigma_fit / alpha_fit) if alpha_fit > 0 else np.nan
    
            mu_err = np.sqrt(pcov[1, 1])
            ok = True
    
        except (RuntimeError, ValueError):
            mu_fit, sigma_fit, A_fit, bkg_fit, mu_err = 0.0, half_width / 4, np.nan, np.nan, 0.0
            alpha_fit = np.nan
            ok = False
            
    
        # shift the section centre onto the true streak centre
        c_corrected = (c[0] + mu_fit * px, c[1] + mu_fit * py)
    
        return c_corrected, sigma_fit, A_fit, mu_fit, ok, mu_err, alpha_fit

    @staticmethod
    def clean_background_pixels(aperture, data, npixels=5, detect_sigma=3.0, 
                                  dilate_size=5, clip_sigma=3.0, maxiters=5):
        """
        Extract pixels from an aperture, mask detected sources, then sigma clip
        the remainder to get a robust background estimate.
        """
        # Get the pixel values and a mask marking which pixels are inside the aperture
        ap_mask = aperture.to_mask(method='center')
        cutout = ap_mask.multiply(data)
        inside = ap_mask.data > 0  # True where pixel is part of the aperture
    
        # Rough noise estimate for source detection threshold (before masking)
        _, bg_median_rough, bg_std_rough = sigma_clipped_stats(
            cutout[inside], sigma=clip_sigma, maxiters=2
        )
        threshold = bg_median_rough + detect_sigma * bg_std_rough
    
        # Detect sources within the cutout
        segm = detect_sources(cutout, threshold, npixels=npixels)
    
        if segm is not None:
            # Dilate the source mask a bit so faint wings get excluded too
            footprint = circular_footprint(radius=dilate_size)
            source_mask = segm.make_source_mask(footprint=footprint)
        else:
            source_mask = np.zeros(cutout.shape, dtype=bool)
    
        # Keep only pixels that are inside the aperture AND not flagged as a source
        good = inside & ~source_mask
        good_pixels = cutout[good]
    
        if good_pixels.size == 0:
            # Fallback: nothing survived masking, use unmasked clipped stats
            good_pixels = cutout[inside]
    
        mean, median, std = sigma_clipped_stats(good_pixels, sigma=clip_sigma, maxiters=maxiters)
        n_used = good_pixels.size
    
        return median, std, n_used, source_mask.sum()

    def photometry_of_streak(self, lengths, centres, theta, stellar_calib_baseline, ratio, zp_error, beta_lsf=None):
        """
        Takes in segments of streak to construct the photometry readings required.
        """
        flux_arcsec_streak = []
        flux_net_list = []
        area_list = []
        area = 0.0

        mu_fits = []
        sigma_fits = []
        A_fits = []
        local_bg_flags = []
        var_flux_net_list = []
        flux_error_list = []
        centres_fit = []
        mu_err_list = []
        
        if beta_lsf is None:
            beta_lsf = np.median(self.moffat_fits[:, 1])
        nu = 2*beta_lsf - 2
        if nu <= 0:
            raise ValueError(f"beta={beta_lsf:.2f} gives nu<=0; LSF undefined.")
    
        def lsf_halfwidth(alpha, EE):
            return alpha * student_t.ppf(0.5 + EE/2, nu) / np.sqrt(nu)
            
        t_list = [] 
        for l, c, t in zip(lengths, centres, theta):
            
            c_fit, sigma_fit, A_fit, mu_fit, fit_ok, mu_err, alpha_fit = self.fit_section_profile(c, t, beta_lsf)
            
            if not fit_ok or abs(mu_fit) > 8 or sigma_fit < 0.5 or sigma_fit > 12 or A_fit <= 0:
                continue
                
            mu_fits.append(mu_fit)
            A_fits.append(A_fit)
            sigma_fits.append(sigma_fit)
            centres_fit.append(c_fit)
            mu_err_list.append(mu_err)
            EE = 0.8              # must equal the EE used in stellar_calibrations
            
            half_w = lsf_halfwidth(alpha_fit, EE)
            width  = 2 * half_w
            
            # background strips at the 90% standoff, scaling with seeing
            inner  = 4 * alpha_fit
            bg_w   = 4 * half_w
            offset = inner + bg_w / 2
            c = c_fit
            on_streak = RectangularAperture(c, l, width, theta=t)
            

            # I will need to implement anti-crossover checks, not hard I dont think I just need to evaluate overlap etc.
            
            dx = -np.sin(t)
            dy =  np.cos(t)
            
            pos_plus  = (c[0] + offset*dx, c[1] + offset*dy)
            pos_minus = (c[0] - offset*dx, c[1] - offset*dy)
            
            off_streak_1 = RectangularAperture(pos_plus, l, width, theta = t)
            off_streak_2 = RectangularAperture(pos_minus, l, width, theta = t)
            
            phot_in = aperture_photometry(self.fits_data, on_streak)
            sum_in = phot_in["aperture_sum"][0]
            arcmin_per_pixel = 1.17 ## Will actually vary given the Fish-eye structure of the lens
            area_in = on_streak.area

            
            bkg1_med, bkg1_std, n1, nmasked1 = self.clean_background_pixels(off_streak_1, self.fits_data)
            bkg2_med, bkg2_std, n2, nmasked2 = self.clean_background_pixels(off_streak_2, self.fits_data)
            
            # Combine the two strips, weighting by number of surviving pixels
            n_total = n1 + n2
            bkg_median = (bkg1_med * n1 + bkg2_med * n2) / n_total
            bkg_std = np.sqrt((bkg1_std**2 * n1 + bkg2_std**2 * n2) / n_total)  # pooled variance
            
            # Optional: flag segments where a large fraction of background got masked out
            frac_masked = (nmasked1 + nmasked2) / (off_streak_1.area + off_streak_2.area)
            local_bg_flags.append(frac_masked > 0.3)  # you already have this list initialized
            
            flux_net = sum_in - bkg_median * area_in
            flux_pix = flux_net / area_in

            flux_net_vignet_correct, lb, ub = self.vignetting_response(flux_net, c[0], c[1])
            
            flux_error = self.flux_error(off_streak_1=off_streak_1, 
                                         off_streak_2=off_streak_2, 
                                         source_flux=flux_net, 
                                         n_pix_source=on_streak.area)

            flux_error_vignet_corrected, lb, ub  = self.vignetting_response(flux_error, c[0], c[1])
            
            flux_arcsec = flux_pix / arcmin_per_pixel
            flux_arcsec_streak.append(flux_arcsec)
            flux_net_list.append(flux_net_vignet_correct)
            area += area_in
            area_list.append(area_in)
            flux_error_list.append(flux_error_vignet_corrected)

        if len(flux_net_list) < 3:
            print(f"Skipping streak: only {len(flux_net_list)} of {len(lengths)} sections "
                  f"passed the profile-fit cuts.")
            return np.nan, np.nan, [], np.nan
        
        # Generate masks to eliminate unwanted readings, these could be due to a number of issues:
        # Background stars shifting the centre line
        # Stars affecting flux 
        # 
        mu_arr = np.asarray(mu_fits)
        A_arr = np.asarray(A_fits)
        sigma_arr = np.asarray(sigma_fits)
        flux_arr = np.asarray(flux_net_list)
        area_arr = np.asarray(area_list)
        flux_err_arr = np.asarray(flux_error_list)
        mu_median = np.median(mu_arr)
        mu_err_arr = np.asarray(mu_err_list)
        mad = np.median(np.abs(mu_arr - mu_median)) * 1.4826
        mu_mask = np.abs(mu_arr - mu_median) < 3 * mad
        centres_arr = np.asarray(centres_fit)
        flux_mask = self.hampel_with_persistence(flux_net_list, window=10, n_sigmas=3, min_run=3)
        flux_mask = np.asarray(flux_mask)
        
        final_mask = mu_mask & flux_mask
        
        self.plotting_light_curve(flux_arr = flux_arr, flux_err = flux_err_arr, mask = final_mask, 
                             centres = centres_arr, segment_length_array = lengths, stellar_calib_baseline = stellar_calib_baseline,
                            ratio = ratio, zp_error = zp_error)
        
        flux_net_streak = flux_arr[final_mask].sum() 
        print(f"Net sum of streak = {flux_net_streak}")
        flux_120_s_normalised = flux_net_streak * (1 / ratio) 
        area = area_arr[final_mask].sum()          # <-- now consistent with the numerator
        
        flux_arcsec = flux_net_streak / area
        mag_arcsec = -2.5*np.log10(flux_arcsec) + stellar_calib_baseline
        mag_total = -2.5*np.log10(flux_net_streak) + stellar_calib_baseline
        mag_total_120_s_normalised = -2.5*np.log10(flux_120_s_normalised) + stellar_calib_baseline
        
        snr_arr = flux_arr / flux_err_arr
        
        return mag_arcsec, mag_total, flux_net_list, mag_total_120_s_normalised

    def hampel_with_persistence(self, y, window, n_sigmas, min_run):
        """
        Flags a point as an outlier only if it deviates AND is not
        corroborated by neighboring points also deviating in the same direction.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        candidate = np.zeros(n, dtype=bool)
        k = 1.4826
    
        # Step 1: standard Hampel candidate flagging
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            local = y[lo:hi]
            med = np.median(local)
            mad = np.median(np.abs(local - med))
            if mad == 0:
                continue
            if np.abs(y[i] - med) > n_sigmas * k * mad:
                candidate[i] = True
    
        # Step 2: require persistence - a "real" deviation shows up in
        # a run of >= min_run consecutive candidates moving the same direction
        mask = np.ones(n, dtype=bool)  # True = keep, False = reject
        i = 0
        while i < n:
            if not candidate[i]:
                i += 1
                continue
            # find the run of consecutive candidate points
            j = i
            while j < n and candidate[j]:
                j += 1
            run_len = j - i
            if run_len < min_run:
                # isolated spike(s) -> genuine outlier(s), reject
                mask[i:j] = False
            # else: sustained deviation -> treat as real signal, keep it
            i = j
    
        return mask

    def local_background_rms(self, fits_data, aperture, sigma_clip_sigma=3.0, maxiters=10):
        """
        Sigma-clipped RMS and Gaussianity check from pixels within a background aperture,
        local to a specific streak segment. This is the primary noise reference used in
        the CCD equation for that segment.
        """
        # 'center' (not 'exact') for noise stats: exact gives flux-weighted partial
        # pixels, which biases std for anything but pure flux summation.
        mask = aperture.to_mask(method='center')
        cutout = mask.multiply(fits_data)
        weights = mask.data
        pixels = cutout[weights > 0]
        pixels = pixels[np.isfinite(pixels)]  # drop NaN/Inf from bad-pixel masks
    
        if pixels.size < 20:
            return np.nan, np.nan, False

        sigma_clip = SigmaClip(sigma=sigma_clip_sigma, maxiters=maxiters)
        clipped = sigma_clip(pixels)
        clean = clipped.compressed()
    
        sigma_std = np.std(clean, ddof=1)
        mad = np.median(np.abs(clean - np.median(clean))) * 1.4826
        
        gaussian_ok = abs(sigma_std - mad) / sigma_std < 0.25 if sigma_std > 0 else False
    
        return sigma_std, mad, gaussian_ok

    def flux_error(self, off_streak_1, off_streak_2, source_flux, n_pix_source):
        
        rms_1, mad_1, ok_1 = self.local_background_rms(self.fits_data, off_streak_1)
        rms_2, mad_2, ok_2 = self.local_background_rms(self.fits_data, off_streak_2)

        sigma_sky = np.sqrt((rms_1**2 + rms_2**2) / 2)

        bkg_term = n_pix_source * sigma_sky**2

        egain = self.header['EGAIN']
        
        shot_term = max(source_flux, 0.0) / egain

        sigma_total = np.sqrt(bkg_term + shot_term)

        return sigma_total

    def plotting_light_curve(self, flux_arr, flux_err, mask, 
                         centres, segment_length_array, stellar_calib_baseline,
                         ratio, zp_error):
    
        number_of_segments = len(flux_arr)
        dt_seg, t_axis, altitudes = self.segment_temporal_lengths(centres, ratio)
        mag_array = np.full(number_of_segments, np.nan)
        mag_err_array = np.full(number_of_segments, np.nan)
        valid_flux = flux_arr > 0
        segment_magnitude_correction = 2.5 * np.log10(dt_seg[valid_flux] / 120)
        mag_array[valid_flux] = (-2.5 * np.log10(flux_arr[valid_flux])+ stellar_calib_baseline+ segment_magnitude_correction)
        
        mag_err_array[valid_flux] = (2.5 / np.log(10)) * (flux_err[valid_flux] / flux_arr[valid_flux])
        
        accepted = mask & valid_flux
        rejected = ~mask & valid_flux
        non_positive = ~valid_flux
            
        fig, ax = plt.subplots(figsize=(7, 5), dpi=500)
        
        ax.errorbar(t_axis[accepted], mag_array[accepted], yerr=mag_err_array[accepted],
                    fmt='o', color='#003E74', ecolor='#002147', alpha=0.8,
                    markersize=4, capsize=2, label='Accepted')
    
        if np.any(rejected):
            ax.scatter(t_axis[rejected], mag_array[rejected], marker='x', color='red',
                       s=60, zorder=5, label='Rejected (quality mask)')
    
        if np.any(non_positive):
            ymin = np.nanmin(mag_array[accepted]) if np.any(accepted) else 0
            ax.scatter(t_axis[non_positive], np.full(non_positive.sum(), ymin - 0.5),
                       marker='x', color='red', s=60)
        
        ax.invert_yaxis()
        ax.set_xlabel('Time (s)', fontsize = 20)
        ax.set_ylabel('Apparent magnitude', fontsize = 20)
        ax.set_title('Satellite brightness variation along streak', fontsize = 20)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # text-only legend entry for the zeropoint systematic (no visible marker/line)
        handles, labels = ax.get_legend_handles_labels()
        zp_handle = Line2D([0], [0], color='none',
                            label=f'Zeropoint systematic: ±{zp_error:.3f} mag')
        handles.append(zp_handle)
        ax.legend(handles=handles)
    
        plt.tight_layout()
        plt.savefig("save_name.png", dpi = 'figure')
        plt.show()

    def _compute_aperture_overlap(self, masks):
        """
        masks: list of photutils ApertureMask objects (from aperture.to_mask()).
        Returns the fraction of total on-streak aperture area that is double-counted
        (summed into more than one segment), plus the coverage array for inspection.
        """
        x0 = min(m.bbox.ixmin for m in masks)
        x1 = max(m.bbox.ixmax for m in masks)
        y0 = min(m.bbox.iymin for m in masks)
        y1 = max(m.bbox.iymax for m in masks)
    
        coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        for m in masks:
            bbox = m.bbox
            sub = coverage[bbox.iymin - y0: bbox.iymax - y0, bbox.ixmin - x0: bbox.ixmax - x0]
            sub += m.data  # fractional (0-1) weights; 'center' method gives ~binary 0/1
    
        nominal_area = sum(m.data.sum() for m in masks)
        overlap_area = np.sum(np.clip(coverage - 1, 0, None))
        overlap_fraction = overlap_area / nominal_area if nominal_area > 0 else 0.0
        return overlap_fraction, coverage

    def photometry_of_streak_simulated(self, lengths, centres, theta, ratio, location, check_overlap=True):
        self.stellar_calibrations(location=location)
    
        flux_net_list = []
        flux_error_list = []
        mu_fits, sigma_fits, A_fits, centres_fit, mu_err_list = [], [], [], [], []
        on_streak_masks = []
    
        def lsf_halfwidth(alpha, EE):
            return alpha * student_t.ppf(0.5 + EE/2, NU) / np.sqrt(NU)
    
        for l, c, t in zip(lengths, centres, theta):
            c_fit, sigma_fit, A_fit, mu_fit, fit_ok, mu_err, alpha_fit = self.fit_section_profile(c, t)
            print(alpha_fit)
            if not fit_ok or abs(mu_fit) > 8 or sigma_fit < 0.5 or sigma_fit > 12 or A_fit <= 0:
                continue
            
            mu_fits.append(mu_fit); A_fits.append(A_fit); sigma_fits.append(sigma_fit)
            centres_fit.append(c_fit); mu_err_list.append(mu_err)
    
            BETA_LSF = 3.5
            NU = 2 * BETA_LSF - 2
            EE = 0.99
            half_w = lsf_halfwidth(alpha_fit, EE)
            width = 2 * half_w
            inner = lsf_halfwidth(alpha_fit, 0.99)
            offset = inner + width / 2
            c = c_fit
    
            on_streak = RectangularAperture(c, l, width, theta=t)
            dx, dy = -np.sin(t), np.cos(t)
            off_streak_1 = RectangularAperture((c[0] + offset*dx, c[1] + offset*dy), l, width, theta=t)
            off_streak_2 = RectangularAperture((c[0] - offset*dx, c[1] - offset*dy), l, width, theta=t)
    
            if check_overlap:
                on_streak_masks.append(on_streak.to_mask(method='center'))
    
            phot_in = aperture_photometry(self.fits_data, on_streak)
            sum_in = phot_in["aperture_sum"][0]
            area_in = on_streak.area
    
            bkg1_med, bkg1_std, n1, nmasked1 = self.clean_background_pixels(off_streak_1, self.fits_data)
            bkg2_med, bkg2_std, n2, nmasked2 = self.clean_background_pixels(off_streak_2, self.fits_data)
            n_total = n1 + n2
            bkg_median = (bkg1_med * n1 + bkg2_med * n2) / n_total
    
            flux_net = sum_in - bkg_median * area_in
            flux_net_vignet_correct, lb, ub = self.vignetting_response(flux_net, c[0], c[1])
    
            flux_error = self.flux_error(off_streak_1, off_streak_2, flux_net, on_streak.area)
            flux_error_vignet_corrected, lb, ub = self.vignetting_response(flux_error, c[0], c[1])
    
            flux_net_list.append(flux_net_vignet_correct)
            flux_error_list.append(flux_error_vignet_corrected)

        if len(flux_net_list) < 3:
            print(f"Skipping streak: only {len(flux_net_list)} of {len(lengths)} sections passed the profile-fit cuts.")
            return np.nan, np.nan, [], np.nan
        
        flux_arr = np.asarray(flux_net_list)
        flux_err_arr = np.asarray(flux_error_list)
        centres_arr = np.asarray(centres_fit)
    
        mu_arr = np.asarray(mu_fits)
        mu_median = np.median(mu_arr)
        mad = np.median(np.abs(mu_arr - mu_median)) * 1.4826
        mu_mask = np.abs(mu_arr - mu_median) < 3 * mad
        flux_mask = np.asarray(self.hampel_with_persistence(flux_net_list, window=10, n_sigmas=3, min_run=3))
        final_mask = mu_mask & flux_mask
    
        dt_seg, t_axis, altitudes = self.segment_temporal_lengths(centres_arr, ratio)
        exposure_time = getattr(self, "exposure_time", 120)
    
        flux_dwell_corrected = np.full(len(flux_arr), np.nan)
        valid = flux_arr > 0
        flux_dwell_corrected[valid] = flux_arr[valid] * (exposure_time / dt_seg[valid])
    
        flux_total = flux_arr[final_mask].sum()
        flux_total_dwell_corrected = np.median(flux_dwell_corrected[final_mask & valid])
    
        overlap_fraction = None
        if check_overlap and on_streak_masks:
            overlap_fraction, coverage = self._compute_aperture_overlap(on_streak_masks)
            print(f"Aperture overlap: {overlap_fraction*100:.2f}% of total on-streak area double-counted")
    
        return {
            "flux_per_section": flux_arr,
            "flux_err_per_section": flux_err_arr,
            "flux_dwell_corrected_per_section": flux_dwell_corrected,
            "mask": final_mask,
            "flux_total": flux_total,
            "flux_total_dwell_corrected": flux_total_dwell_corrected,
            "aperture_overlap_fraction": overlap_fraction,
        }
