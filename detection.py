"""Background removal, Hough/blob streak detection and centreline refinement."""
import cv2
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from skimage.measure import ransac
from scipy.ndimage import median_filter

# Method bodies are copied unchanged from the original single class.
# They still use self, so they only work as part of OperatingFitsFiles (frame.py).

class DetectionMixin:
    def get_centerline(self, x, y, n_bins = 100):
        """
        Solution provided from Claude
        """
        pts = np.column_stack([x, y])
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        principal = vt[0]                      # dominant direction of the trail
        proj = (pts - mean) @ principal        # 1-D coordinate along the trail
    
        order = np.argsort(proj)
        proj_sorted = proj[order]
    
        bins = np.linspace(proj_sorted.min(), proj_sorted.max(), n_bins + 1)
        centers_x, centers_y = [], []
        for i in range(n_bins):
            sel = (proj >= bins[i]) & (proj < bins[i+1])
            if sel.sum() > 0:
                centers_x.append(x[sel].mean())
                centers_y.append(y[sel].mean())
    
        return np.array(centers_x), np.array(centers_y)

    def remove_background(self, img, sigma=3, maxiters=10, kernel_size=(70, 70), filter_size=(3, 3)):
        sigma_clip = SigmaClip(sigma, maxiters=maxiters)
        bkg = Background2D(
            img, kernel_size, filter_size=filter_size,
            sigma_clip=sigma_clip, bkg_estimator=MedianBackground(),
        )
        return img - bkg.background

    def preprocess(self, img, brightness_cuts=(1, 1), thresholding_cut=0.5,
                   flux_prop_thresholds=(0.1, 0.2, 0.3, 1.0), blur_kernel_sizes=(3, 5, 9, 11),
                   canny_thresholds=(0, 200)):
        img = self.remove_background(img.astype(np.float64))
    
        up_limit = img.mean() + brightness_cuts[1] * img.std()
        low_limit = img.mean() - brightness_cuts[0] * img.std()
        img = np.clip(img, None, up_limit)
        img[img <= low_limit] = 0
    
        norm = (img - img.mean()) / (img.std() + 1e-9)
        norm -= norm.min()
        norm = 255 * (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
    
        limit = norm.mean() + thresholding_cut * norm.std()
        _, thresholded = cv2.threshold(norm.astype(np.float32), limit, 255, cv2.THRESH_BINARY)
        thresholded = cv2.convertScaleAbs(thresholded)
    
        prop_bright = np.mean(thresholded > np.mean(thresholded))
        kernel = blur_kernel_sizes[-1]
        for thresh_frac, k in zip(flux_prop_thresholds, blur_kernel_sizes):
            if prop_bright < thresh_frac:
                kernel = k
                break
    
        blurred = cv2.medianBlur(thresholded, kernel)
        edges = cv2.Canny(blurred, *canny_thresholds)
    
        return thresholded, blurred, edges

    def detect_segments(self, edges, rho=1, theta_deg=0.5, hough_threshold=50,
                         min_line_length=200, max_line_gap=10):
        theta = np.radians(theta_deg)
        segments = cv2.HoughLinesP(
            edges, rho, theta, hough_threshold,
            minLineLength=min_line_length, maxLineGap=max_line_gap,
        )
        if segments is None:
            return np.empty((0, 4))
        return segments.reshape(-1, 4).astype(np.float64)

    def detect_blob_lines(self, thresholded, min_area=100, min_elongation=5):
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        segments = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            (cx, cy), (w, h), angle = cv2.minAreaRect(c)
            length, width = max(w, h), max(min(w, h), 1e-3)
            if length / width < min_elongation:
                continue  # too round/blob-like to be a streak (e.g. a star)
    
            vx, vy, x0, y0 = cv2.fitLine(c.reshape(-1, 2).astype(np.float32),
                                          cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            half_len = length / 2
            segments.append((x0 - vx * half_len, y0 - vy * half_len,
                              x0 + vx * half_len, y0 + vy * half_len))
    
        return np.array(segments) if segments else np.empty((0, 4))

    def segment_to_hough_params(self, x1, y1, x2, y2):
        line_angle = np.arctan2(y2 - y1, x2 - x1)
        theta_n = (line_angle - np.pi / 2) % np.pi
        dist = x1 * np.cos(theta_n) + y1 * np.sin(theta_n)
        return dist, theta_n

    def merge_duplicate_segments(self, segments, dist_tol=15, angle_tol_deg=5):
        if len(segments) == 0:
            return segments
    
        params = np.array([self.segment_to_hough_params(*s) for s in segments])
        used = np.zeros(len(segments), dtype=bool)
        merged = []
    
        for i in range(len(segments)):
            if used[i]:
                continue
            group_idx = [i]
            used[i] = True
            for j in range(i + 1, len(segments)):
                if used[j]:
                    continue
                d_diff = abs(params[i, 0] - params[j, 0])
                a_diff = abs(params[i, 1] - params[j, 1])
                a_diff = min(a_diff, np.pi - a_diff)
                if d_diff < dist_tol and np.degrees(a_diff) < angle_tol_deg:
                    group_idx.append(j)
                    used[j] = True
    
            group = segments[group_idx]
            lengths = np.hypot(group[:, 2] - group[:, 0], group[:, 3] - group[:, 1])
            merged.append(group[np.argmax(lengths)])
    
        return np.array(merged)

    def detect_and_convert(self, undistorted_fits_path,
                            preprocess_kwargs=None, hough_kwargs=None,
                            merge_kwargs=None, n_samples_per_segment=200):
    
        with fits.open(undistorted_fits_path, memmap=False) as hdul:
            undistorted_data = hdul[0].data.copy()
            header = hdul[0].header.copy()
    
        ra_centre_deg = header["UND_RA0"]
        dec_centre_deg = header["UND_DEC0"]
        half_tan = header["UND_HTAN"]
        out_width = header["UND_W"]
        out_height = header["UND_H"]
        crop_x0 = header.get("CROP_X0", 0)
        crop_y0 = header.get("CROP_Y0", 0)
        obs_time_tuple = self.get_obs_time_tuple()
    
        preprocess_kwargs = preprocess_kwargs or {}
        hough_kwargs = hough_kwargs or {}
        merge_kwargs = merge_kwargs or {}
    
        thresholded, blurred, edges = self.preprocess(undistorted_data, **preprocess_kwargs)
    
        raw_segments_hough = self.detect_segments(edges, **hough_kwargs)
        raw_segments_blob = self.detect_blob_lines(thresholded)
        raw_segments = np.vstack([raw_segments_hough, raw_segments_blob])
        print(f"HoughLinesP found {len(raw_segments_hough)}, blob detector found "
              f"{len(raw_segments_blob)} raw segment(s).")
    
        segments = self.merge_duplicate_segments(raw_segments, **merge_kwargs)
        print(f"After merging near-duplicates: {len(segments)} streak(s).")
    
        results = []
        for i, (x1, y1, x2, y2) in enumerate(segments):
            x_seg = np.linspace(x1, x2, n_samples_per_segment)
            y_seg = np.linspace(y1, y2, n_samples_per_segment)
    
            x_crop_dist, y_crop_dist = self.undistorted_xy_to_original_xy(
                x_seg, y_seg, out_width, out_height, half_tan,
                ra_centre_deg, dec_centre_deg, obs_time_tuple,
                crop_x0=crop_x0, crop_y0=crop_y0,
            )
            x_original_dist = x_crop_dist + crop_x0
            y_original_dist = y_crop_dist + crop_y0
            length_px = np.hypot(x2 - x1, y2 - y1)

            distance = np.sqrt((x2 - 3246)**2+(y2 - 3246)**2) # HARDCODED PLEASE CHANGE TO IMPROVE FUNCTIONALITY OSCAR!!!!
            
            if length_px > 200:
                if distance < 3000:
                    print(f"  Streak {i}: undistorted endpoints ({x1:.0f},{y1:.0f}) -> "
                          f"({x2:.0f},{y2:.0f}), length ~{length_px:.1f} px")
                        
                    results.append({
                        "undistorted_endpoints": (x1, y1, x2, y2),
                        "undistorted_xy": (x_seg, y_seg),
                        "cropped_frame_xy": (x_crop_dist, y_crop_dist),
                        "original_frame_xy": (x_original_dist, y_original_dist),
                    })
        return results, {"thresholded": thresholded, "blurred": blurred, "edges": edges}

    def refine_streak_centerline(self, x_seg, y_seg, image=None,
                              perp_halfwidth=4, along_step=5.0,
                              fit_method='gaussian',  # 'gaussian' | 'epsf' | 'centroid'
                              epsf=None, mad_thresh=3.0,
                              min_snr=2.0):
        """
        Refine a coarse Hough-derived streak polyline (x_seg, y_seg) into a
        sub-pixel-accurate centerline, by sampling perpendicular cross-sections
        of the RAW (undistorted-projection-free) image data along the prior
        path and centroiding each one.
        
        Parameters
        ----------
        x_seg, y_seg : array-like
            Coarse path in ORIGINAL (distorted) frame pixel coords -- e.g.
            results[i]["original_frame_xy"] after the crop_x0/crop_y0 fix.
        image : 2D array, optional
            Background-subtracted image to sample. Defaults to
            self.remove_background(self.fits_data).
        perp_halfwidth : float
            Half-width (px) of the perpendicular search/sampling window.
            Keep this tight (a few px) -- this is your main defense against
            the fit wandering onto a nearby star.
        along_step : float
            Spacing (px) between along-track sampling points. Finer than 1 px
            is fine since you're just resampling a smooth prior line.
        use_gaussian : bool
            If True, fit a 1D Gaussian to each cross-section for sub-pixel
            centroid + amplitude + width. Falls back to flux-weighted centroid
            if the fit fails to converge.
        mad_thresh : float
            Outlier rejection threshold (multiples of MAD) on perpendicular
            offset from the smooth prior -- flags likely star contamination.
        min_snr : float
            Minimum peak amplitude / local noise ratio to trust a cross-section
            at all; below this it's treated as background/gap and interpolated.
    
        Returns
        -------
        dict with:
            'x_refined', 'y_refined'   : sub-pixel centerline (bad points
                                          interpolated from good neighbors)
            'amplitude'                : per-step fitted/measured peak flux
            'width'                    : per-step fitted Gaussian sigma (nan
                                          if use_gaussian=False or fit failed)
            'flagged'                  : boolean mask, True where a point was
                                          rejected as an outlier / low-SNR and
                                          had to be interpolated
            'perp_offset_raw'          : raw perpendicular offset (px) from the
                                          prior line at each step, pre-rejection
        """
        if image is None:
            image = self.remove_background(self.fits_data.astype(np.float64))
    
        x_seg = np.asarray(x_seg, dtype=np.float64)
        y_seg = np.asarray(y_seg, dtype=np.float64)
    
        # --- 1. Resample the coarse prior to even along-track spacing ---
        dx = np.diff(x_seg)
        dy = np.diff(y_seg)
        seg_lengths = np.hypot(dx, dy)
        cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_len = cum_len[-1]
        if total_len < along_step:
            raise ValueError("Streak segment too short to refine.")
    
        n_steps = max(int(total_len / along_step), 2)
        s_new = np.linspace(0, total_len, n_steps)
        x_prior = np.interp(s_new, cum_len, x_seg)
        y_prior = np.interp(s_new, cum_len, y_seg)
    
        # local tangent direction at each point (finite difference, smoothed)
        tx = np.gradient(x_prior)
        ty = np.gradient(y_prior)
        tnorm = np.hypot(tx, ty)
        tnorm[tnorm == 0] = 1.0
        tx, ty = tx / tnorm, ty / tnorm
        # perpendicular unit vector
        px_, py_ = -ty, tx
    
        # --- 2. Sample perpendicular cross-sections and centroid each one ---
        perp_offsets = np.linspace(-perp_halfwidth, perp_halfwidth, int(4 * perp_halfwidth) + 1)
    
        x_refined = np.full(n_steps, np.nan)
        y_refined = np.full(n_steps, np.nan)
        amplitude = np.full(n_steps, np.nan)
        width = np.full(n_steps, np.nan)
        perp_offset_raw = np.full(n_steps, np.nan)
    
        def gauss1d(u, amp, mu, sigma, offset):
            return offset + amp * np.exp(-0.5 * ((u - mu) / sigma) ** 2)
    
        for i in range(n_steps):
            xs = x_prior[i] + perp_offsets * px_[i]
            ys = y_prior[i] + perp_offsets * py_[i]
            profile = map_coordinates(image, [ys, xs], order=1, mode='nearest')

            local_noise = np.std(profile) + 1e-6
            peak = profile.max()

            if peak / local_noise < min_snr:
                continue

            if fit_method == 'gaussian':
                try:
                    p0 = [peak - profile.min(), perp_offsets[np.argmax(profile)],
                          max(perp_halfwidth / 3, 1.0), np.median(profile)]
                    popt, _ = curve_fit(gauss1d, perp_offsets, profile, p0=p0, maxfev=1000)
                    amp_fit, mu_fit, sigma_fit, _ = popt
                    if not (-perp_halfwidth <= mu_fit <= perp_halfwidth) or sigma_fit <= 0:
                        raise RuntimeError("Fit landed outside window / degenerate.")
                except Exception:
                    w = profile - profile.min()
                    mu_fit = np.sum(perp_offsets * w) / (np.sum(w) + 1e-9)
                    amp_fit = peak
                    sigma_fit = np.nan
                    
            else:  # 'centroid'
                w = profile - profile.min()
                mu_fit = np.sum(perp_offsets * w) / (np.sum(w) + 1e-9)
                amp_fit = peak
                sigma_fit = np.nan

            x_refined[i] = x_prior[i] + mu_fit * px_[i]
            y_refined[i] = y_prior[i] + mu_fit * py_[i]
            amplitude[i] = amp_fit
            width[i] = sigma_fit
            perp_offset_raw[i] = mu_fit

        # --- 3. Outlier rejection: reject points whose perpendicular offset ---
        #     deviates too far from the smooth local trend (catches stars/glitches)
        flagged = np.isnan(x_refined)  # already-failed (low SNR) points
    
        valid = ~np.isnan(perp_offset_raw)
        if valid.sum() >= 5:
            # smooth trend via median filter over a small window
            
            trend = np.full(n_steps, np.nan)
            trend[valid] = median_filter(perp_offset_raw[valid], size=min(9, valid.sum()), mode='nearest')
            resid = np.abs(perp_offset_raw - trend)
            mad = np.nanmedian(np.abs(resid[valid] - np.nanmedian(resid[valid]))) + 1e-9
            outliers = valid & (resid > mad_thresh * 1.4826 * mad)
            flagged |= outliers
            x_refined[outliers] = np.nan
            y_refined[outliers] = np.nan
    
        # --- 4. Interpolate over flagged/NaN points so the centerline stays continuous ---
        good = ~np.isnan(x_refined)
        if good.sum() < 2:
            raise RuntimeError("Refinement failed -- too few good cross-sections "
                                "(streak too faint, or entirely contaminated).")
    
        idx = np.arange(n_steps)
        x_refined = np.interp(idx, idx[good], x_refined[good])
        y_refined = np.interp(idx, idx[good], y_refined[good])
    
        return {
            "x_refined": x_refined,
            "y_refined": y_refined,
            "amplitude": amplitude,
            "width": width,
            "flagged": flagged,
            "perp_offset_raw": perp_offset_raw,
            "s": s_new,  # along-track distance, useful for plotting flux vs position
        }

    def split_streak_into_sections(self, x, y, step = 20):
        """
        Given ordered x, y coordinates along a curve, return points spaced
        approximately `step` pixels apart along the path (arc length).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        dx = np.diff(x)
        dy = np.diff(y)
        seg_lengths = np.sqrt(dx**2 + dy**2)
        arc = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = arc[-1]
    
        # Arc-length positions of segment boundaries
        boundaries_arc = np.arange(0, total_length, step)
        if boundaries_arc[-1] != total_length:
            boundaries_arc = np.append(boundaries_arc, total_length)
    
        # Interpolate x, y at each boundary
        x_bound = np.interp(boundaries_arc, arc, x)
        y_bound = np.interp(boundaries_arc, arc, y)
    
        starts = np.column_stack([x_bound[:-1], y_bound[:-1]])
        ends   = np.column_stack([x_bound[1:],  y_bound[1:]])
        lengths = np.sqrt(np.sum((ends - starts)**2, axis=1))
        
        centres = (starts + ends) // 2
        dx_s, dy_s = (ends - starts).T
        theta = np.arctan2(dy_s, dx_s)
        
        return lengths, centres, theta
