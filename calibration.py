"""Stellar Moffat fits, enclosed-energy radius, vignetting, extinction and zeropoint."""
from math import cos, log10, sqrt

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.centroids import centroid_2dg, centroid_sources
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from scipy.constants import c, h
from scipy.constants import k as k_B
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

# Method bodies are copied unchanged from the original single class.
# They still use self, so they only work as part of OperatingFitsFiles (frame.py).


class CalibrationMixin:
    def vignetting_response(self, flux, x, y):
        """
        Corrects the vignetting of any selected pixel within the image field.
        """
        vignetting_coefficient = 0.000204
        vignetting_coefficient_error = 0.000015
        radius = sqrt((float(x) - 4375)**2 + (float(y)- 4375)**2)
        
        vignetted_intensity = flux / (cos(vignetting_coefficient * radius)**4)
        
        vignetted_intensity_error_lb = flux / (cos((vignetting_coefficient + vignetting_coefficient_error) * radius)**4)
        vignetted_intensity_error_ub = flux / (cos((vignetting_coefficient - vignetting_coefficient_error) * radius)**4)

        error_lb = (vignetted_intensity - vignetted_intensity_error_lb) / vignetted_intensity * 100
        error_ub = (vignetted_intensity - vignetted_intensity_error_ub) / vignetted_intensity * 100
        
        return vignetted_intensity, error_lb, error_ub

    def photometry_analysis(self, x, y, source_radius=None,
                        annulus_inner=None, annulus_outer=None):
        # print(source_radius)
        if source_radius is None:
            source_radius = 1.5 * self.median_fwhm
        if annulus_inner is None:
            annulus_inner = 4.0 * self.median_fwhm
        if annulus_outer is None:
            annulus_outer = annulus_inner + 4.0 * self.median_fwhm
    
        central_aperture = CircularAperture((x, y), source_radius)
        annulus_aperture = CircularAnnulus((x, y), annulus_inner, annulus_outer)
    
        cen_pho = aperture_photometry(self.fits_data, central_aperture)
    
        annulus_mask = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_mask.multiply(self.fits_data)
        annulus_data_1d = annulus_data[annulus_mask.data > 0]
        _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d)
    
        bkg_total_in_source_aperture = bkg_median * central_aperture.area
        return cen_pho['aperture_sum'][0] - bkg_total_in_source_aperture

    @staticmethod
    def kasten_formula(zenith_angle):
        return (np.cos(np.radians(zenith_angle)) + 0.50572*(96.07995 - zenith_angle)**(-1.6364))**(-1)

    # Pecaut & Mamajek (2013), ApJS 208, 9 -- representative dwarf sequence anchor points   
    @staticmethod
    def bv_to_teff(bv, _PM_BV, _PM_TEFF):
        bv_clamped = np.clip(bv, _PM_BV.min(), _PM_BV.max())
        return np.interp(bv_clamped, _PM_BV, _PM_TEFF)

    def extinction_magnitude_correction(self, zenith_angle, vmag, bmag):
        # Paranal extinction values obtained by Patat et al. 2011
        # https://www.aanda.org/articles/aa/full_html/2011/03/aa15537-10/aa15537-10.html
        # We note that between 6775-7000 we have interpolated.
        _PM_BV   = np.array([-0.33, -0.30, -0.24, -0.17, -0.11, -0.02, 0.00, 0.15,
                           0.30,  0.44,  0.58,  0.68,  0.82,  0.92,  1.15, 1.40, 1.80])
        _PM_TEFF = np.array([42000, 30000, 20000, 15700, 12500, 10000, 9700, 8200,
                           7200,  6650,  6050,  5770,  5250,  4900,  4350, 3800, 3200])
        
        wavelength_A = np.array([
            4025, 4075, 4125, 4175, 4225, 4275, 4325, 4375, 4425, 4475,
            4525, 4575, 4625, 4675, 4725, 4775, 4825, 4875, 4925, 4975,
            5025, 5075, 5125, 5175, 5225, 5275, 5325, 5375, 5425, 5475,
            5525, 5575, 5625, 5675, 5725, 5775, 5825, 5875, 5925, 5975,
            6025, 6075, 6125, 6175, 6225, 6275, 6325, 6375, 6425, 6475,
            6525, 6575, 6625, 6675, 6725, 6775, 7000
        ])
    
        k_lambda = np.array([
            0.330, 0.316, 0.298, 0.285, 0.274, 0.265, 0.253, 0.241, 0.229, 0.221,
            0.212, 0.204, 0.198, 0.190, 0.185, 0.182, 0.176, 0.169, 0.162, 0.157,
            0.156, 0.153, 0.146, 0.143, 0.141, 0.139, 0.139, 0.134, 0.133, 0.131,
            0.129, 0.127, 0.128, 0.130, 0.134, 0.132, 0.124, 0.122, 0.125, 0.122,
            0.117, 0.115, 0.108, 0.104, 0.102, 0.099, 0.095, 0.092, 0.085, 0.086,
            0.083, 0.081, 0.076, 0.072, 0.068, 0.064, 0.064
        ])
    
        sigma_k = np.array([
            0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.003, 0.003, 0.003,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.003,
            0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002
        ])
    
        # QE values obtained for an adjacent sensor (ours does not have QE curve data available!)
        transmission_filter = 0.95
        qe_wavelength_nm = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700])
        qe_percent = np.array([64, 74, 78, 80, 80, 78, 72, 67, 61, 55, 48, 43, 37]) * transmission_filter * 0.01
        qe_wavelength_A = qe_wavelength_nm * 10
    
        lam_min = max(wavelength_A.min(), qe_wavelength_A.min())
        lam_max = min(wavelength_A.max(), qe_wavelength_A.max())
        lam_common = np.arange(lam_min, lam_max + 1, 5)
    
        k_interp     = CubicSpline(wavelength_A, k_lambda)(lam_common)
        sigma_interp = CubicSpline(wavelength_A, sigma_k)(lam_common)
        S_interp     = np.interp(lam_common, qe_wavelength_A, qe_percent)
    
        # B-V colour -> approximate effective temperature (Ballesteros 2012)
        bv = bmag - vmag
        T_eff = self.bv_to_teff(bv, _PM_BV, _PM_TEFF)
        
        lam_m = lam_common * 1e-10
        planck = (2 * h * c**2) / (lam_m**5 * (np.exp((h * c) / (lam_m * k_B * T_eff)) - 1))
        weight = S_interp * planck

        # print(f"vmag={vmag:.2f} bmag={bmag:.2f} B-V={bv:.3f} T_eff={T_eff:.1f}")
        numerator   = simpson(k_interp * weight, x=lam_common)
        denominator = simpson(weight, x=lam_common)
        extinction_coeff = numerator / denominator
    
        var_num = simpson((sigma_interp**2) * (S_interp**2), x=lam_common)
        sigma_extinction_coeff = np.sqrt(var_num) / denominator
    
        X = self.kasten_formula(zenith_angle)
        m_correction = X * extinction_coeff
    
        return m_correction

    @staticmethod
    def _r_ee(alpha, beta, EE):
        """2D circular Moffat enclosed-energy radius."""
        return alpha * np.sqrt((1 - EE)**(1/(1 - beta)) - 1)

    @staticmethod
    def _hw_ee(alpha, beta, EE):
        """Cross-trail enclosed-energy halfwidth (infinite-line limit; valid for L >> alpha)."""
        nu = 2*beta - 2
        return alpha * student_t.ppf(0.5 + EE/2, nu) / np.sqrt(nu)

    def fit_moffat(self, xc, yc, box=10):
        """Free alpha and beta. Returns (alpha, beta) or None."""
        y0, x0 = int(round(yc)), int(round(xc))
        cut = self.fits_data[y0-box:y0+box+1, x0-box:x0+box+1].astype(float)
        if cut.shape != (2*box+1, 2*box+1):
            return None
        yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
        cx, cy = xc - x0 + box, yc - y0 + box
    
        def model(coords, amp, mx, my, alpha, beta, bkg):
            x, y = coords
            return (amp*(1 + ((x-mx)**2 + (y-my)**2)/alpha**2)**(-beta) + bkg).ravel()
    
        try:
            p, _ = curve_fit(model, (xx, yy), cut.ravel(),
                             p0=[cut.max()-np.median(cut), cx, cy, 3.0, 3.0, np.median(cut)],
                             sigma=np.sqrt(np.maximum(cut, 1)).ravel(),
                             bounds=([0, cx-3, cy-3, 0.5, 1.5, -np.inf],
                                     [np.inf, cx+3, cy+3, 20.0, 15.0, np.inf]),
                             maxfev=20000)
        except (RuntimeError, ValueError):
            return None
        return p[3], p[4]

    def ee_halfwidth_streak(self, EE=0.80):
        a, b = self.moffat_fits[:, 0], self.moffat_fits[:, 1]
        _, med, _ = sigma_clipped_stats(self._hw_ee(a, b, EE))
        return med

    def ee_radius_scan(self, stars, radii=None, r_in=None, r_out=None):
        """
        Empirical aperture scan. Minimises the scatter in zeropoint across stars,
        which has a real minimum (unlike normalised EE, which trivially goes to zero
        at the normalisation radius).
    
        stars: list of (xc, yc, vmag, mag_correction).
        Returns (r_best, sd_best_mag, radii, sd_mag).
        """
        a_med = np.median(self.moffat_fits[:, 0])
        if radii is None:
            radii = np.arange(2.0, 8.0*a_med, 0.5)
        if r_in is None:
            r_in = 5.0 * a_med
        if r_out is None:
            r_out = 8.0 * a_med
    
        rows = []
        for xc, yc, vmag, mag_corr in stars:
            f = np.array([self.photometry_analysis(xc, yc, r, r_in, r_out) for r in radii])
            with np.errstate(invalid="ignore", divide="ignore"):
                zp = vmag + 2.5*np.log10(np.where(f > 0, f, np.nan)) + mag_corr
            rows.append(zp)
    
        zp = np.array(rows)
        sd = np.nanstd(zp, axis=0, ddof=1)
        ok = np.isfinite(sd) & (np.sum(np.isfinite(zp), axis=0) >= 3)
        if not ok.any():
            return np.nan, np.nan, radii, sd
        i = np.flatnonzero(ok)[np.nanargmin(sd[ok])]
        return radii[i], sd[i], radii, sd

    def plot_stars(self, star_centres, r_star, r_in, r_out, box=50):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    
        n = len(star_centres)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        axes = np.atleast_1d(axes).ravel()
    
        h = box // 2
        for ax, (xc, yc) in zip(axes, star_centres):
            x0, y0 = int(round(xc)) - h, int(round(yc)) - h
            cut = self.fits_data[y0:y0+box, x0:x0+box]
            if cut.shape != (box, box):
                ax.axis("off")
                continue
            vmin, vmax = np.percentile(cut, [5, 99])
            ax.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                      extent=[x0-0.5, x0+box-0.5, y0-0.5, y0+box-0.5])
            for r, c in ((r_star, "lime"), (r_in, "cyan"), (r_out, "cyan")):
                ax.add_patch(Circle((xc, yc), r, fill=False, color=c, lw=1))
            ax.set_title(f"({xc:.0f}, {yc:.0f})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    
        for ax in axes[n:]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def stellar_calibrations(self, location, n_refs=10, max_sep_px=3000, EE=0.80):
        """
        location: (ra, dec) in degrees for the target star/streak.
        n_refs: max number of nearby calibration stars to use.
        max_sep_px: don't use a calibration star farther than this from the target,
                    even if fewer than n_refs are found.
        EE: enclosed-energy fraction defining the photometric aperture. The streak
            halfwidth must use the same fraction so the aperture correction cancels.
        """
        centroid_func = centroid_2dg
    
        ra_loc, dec_loc = location
        x_loc_arr, y_loc_arr = raDecToXYPP(
            np.atleast_1d(float(ra_loc)), np.atleast_1d(float(dec_loc)),
            self.exposure_start_jd, self.pp
        )
        x_loc, y_loc = x_loc_arr[0], y_loc_arr[0]
    
        candidates = []
        with open("stellar_calibration1.txt") as f:
            for line in f:
                star_id, ra, dec, vmag, bmag, sg = line.strip().split(",")
    
                ra_arr = np.atleast_1d(float(ra))
                dec_arr = np.atleast_1d(float(dec))
                x_pred, y_pred = raDecToXYPP(ra_arr, dec_arr,
                                             self.exposure_start_jd, self.pp)
                x, y = x_pred[0], y_pred[0]
    
                radius = sqrt((x - 4375)**2 + (y - 4375)**2)
                if radius >= 3500:
                    continue
    
                sep = sqrt((x - x_loc)**2 + (y - y_loc)**2)
                if sep > max_sep_px:
                    continue
    
                altaz_frame = AltAz(obstime=self.exposure_start, location=self.location)
                star_coords = SkyCoord(ra=ra_arr, dec=dec_arr, unit=(u.deg, u.deg))
                star_altaz = star_coords.transform_to(altaz_frame)
                zenith_angle = 90 - star_altaz.alt.deg[0]
    
                mag_correction = self.extinction_magnitude_correction(
                    float(zenith_angle), float(vmag), float(bmag)
                )
                candidates.append((sep, star_id, x, y, float(vmag),
                                   float(mag_correction), float(zenith_angle)))
    
        if not candidates:
            raise ValueError(
                f"No calibration stars found within {max_sep_px}px of target "
                f"and inside the vignetting-safe radius."
            )
    
        candidates.sort(key=lambda c: c[0])
        selected = candidates[:n_refs]
    
        # --- pass 1: centroid + Moffat fit (alpha and beta both free) ---
        centroids = []
        fits = []
        for sep, star_id, x, y, vmag, mag_correction, zenith_angle in selected:
            x_ref, y_ref = centroid_sources(self.fits_data, x, y,
                                            box_size=39, centroid_func=centroid_func)
            xc, yc = x_ref[0], y_ref[0]
            centroids.append((xc, yc, vmag, mag_correction, star_id, sep))
            
            r_field = np.hypot(xc - 4375, yc - 4375)
            pa_radial = np.arctan2(yc - 4375, xc - 4375)
    
            ab = self.fit_moffat(xc, yc)
            
            alpha, beta = ab
            fits.append((alpha, beta, r_field))
            #print(f"{star_id}  alpha={alpha:.2f}  beta={beta:.2f}")
    
        if len(fits) < 3:
            raise ValueError(f"Only {len(fits)} usable Moffat fits for this streak.")
    
        self.moffat_fits = np.array(fits)                    # (alpha, beta, r_field)
    
        r_all = self._r_ee(self.moffat_fits[:, 0], self.moffat_fits[:, 1], EE)
        _, r_star, r_sd = sigma_clipped_stats(r_all)
        self.r_star_err = r_sd / np.sqrt(len(r_all))
        self.streak_halfwidth = self.ee_halfwidth_streak(EE=EE)
    
        a_med = np.median(self.moffat_fits[:, 0])
        r_in  = 5.0 * a_med                                  # tied to alpha, not to r_star
        r_out = 8.0 * a_med
        # print(f"r_star={r_star:.2f} +/- {self.r_star_err:.2f} px  "
        #       f"streak halfwidth={self.streak_halfwidth:.2f} px  "
        #       f"annulus {r_in:.1f}-{r_out:.1f} px")
    
        # --- pass 2: photometry + zeropoints ---
        zero_points = []
        star_centres = []
        scan_stars = []
        for xc, yc, vmag, mag_correction, star_id, sep in centroids:
            flux = self.photometry_analysis(xc, yc, r_star, r_in, r_out)
            vignetted_flux, lb, ub = self.vignetting_response(flux, xc, yc)
            if not np.isfinite(vignetted_flux) or vignetted_flux <= 0:
                # print(f"skipping {star_id}: flux={vignetted_flux}")
                continue
            star_centres.append((xc, yc))
            scan_stars.append((xc, yc, vmag, mag_correction))
            zero_point = vmag + 2.5 * log10(vignetted_flux)
            zero_point_corrected = zero_point + mag_correction
            print(f"zp_corrected={zero_point_corrected:.4f}  zp={zero_point:.4f}  {star_id}  "
                  f"vmag={vmag}  sep_px={sep:.1f}  flux={vignetted_flux:.1f}")
            zero_points.append(zero_point_corrected)
    
        if len(zero_points) < 3:
            print(f"Warning: only {len(zero_points)} reference stars used — "
                  f"sigma clipping may not be meaningful.")
    
        r_best, sd_best, _, _ = self.ee_radius_scan(scan_stars, r_in=r_in, r_out=r_out)
        print(f"empirical optimum {r_best:.1f} px (zp scatter {sd_best:.4f} mag) "
              f"vs model r_star={r_star:.1f}")
    
        mean, median, std = sigma_clipped_stats(np.array(zero_points), sigma=3.0, maxiters=5)
        self.plot_stars(star_centres, r_star, r_in, r_out)
        return mean, median, std, star_centres
