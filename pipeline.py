"""Per-streak driver: refine, calibrate, match, measure and plot."""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import jd2Date
from scipy.interpolate import splev, splprep

# Method bodies are copied unchanged from the original single class.
# They still use self, so they only work as part of OperatingFitsFiles (frame.py).


class PipelineMixin:
    def run_hough_transform(self):
        UNDISTORTED_FITS = f"{self.file_name}_undistorted.fits"
        CROPPED_FITS = f"{self.file_name}_cropped{self.cutoff_deg}deg.fits"
        
        results, intermediates = self.detect_and_convert(
            undistorted_fits_path=UNDISTORTED_FITS,
            hough_kwargs=dict(rho=1, theta_deg=0.5, hough_threshold=50,
                               min_line_length=200, max_line_gap=10),
            merge_kwargs=dict(dist_tol=40, angle_tol_deg=5),
        )
        print(f"\n{len(results)} streak(s) with recovered segments and distorted-frame paths.")
    
        with fits.open(UNDISTORTED_FITS, memmap=False) as hdul:
            undistorted_data = hdul[0].data.copy()
    
        with fits.open(CROPPED_FITS, memmap=False) as hdul:
            cropped_original_data = hdul[0].data.copy()

        if self.simulated:
            self.analyse_simulated_streak(
            results,
            self.ra_truth_start, self.dec_truth_start,
            self.ra_truth_end, self.dec_truth_end,
            original_data=self.fits_data
            )
        else:
            self.plot_streaks(undistorted_data, results, original_data=self.fits_data)

    # The following code needs some serious work!
    def plot_streaks(self, undistorted_data, results, original_data=None):
        if original_data is not None:
            i = 0
            for r in results:
                refined = self.refine_streak_centerline(
                *r["original_frame_xy"],   # x_seg, y_seg (after your crop_x0 fix)
                perp_halfwidth=10,
                    
                )
                
                x_seg, y_seg = refined['x_refined'], refined['y_refined']
                #inliers, model = self.ransac_streak_spline_fitting(points = points, order=2)
                cx, cy = self.get_centerline(x_seg, y_seg, n_bins=40)
                tck, u = splprep([cx, cy], s=len(cx) * 10, k=2)
                
                u_check = np.linspace(0, 1, 500)
                spline_xs, spline_ys = splev(u_check, tck)

                x_mid, y_mid = splev(0.5, tck)
                
                _, ra_c, dec_c, _ = xyToRaDecPP(
                [jd2Date(self.exposure_start_jd)], [x_mid], [y_mid], [1], self.pp
                )

                spline_xy = np.column_stack([spline_xs, spline_ys])
                print(ra_c, dec_c)

                try:
                    satellites = self.streak_passes(float(ra_c[0]), float(dec_c[0]), radius = 5)
                except Exception as e:
                    print("Streak passes error, there appears to be no data for that query.")
                    satellites = {}

                    
                mean, median, std, star_centres = self.stellar_calibrations(location = (ra_c, dec_c))
                
                candidates = []
                
                lengths, centres, theta = self.split_streak_into_sections(spline_xs, 
                                                                          spline_ys)
                
                for sat_name, positions in satellites.items():
                    sat_xs, sat_ys, times = [], [], []
                
                    for pos in positions:
                        x_arr, y_arr, ra, dec, jd, range_km, norad_id = pos
                        
                        if range_km < 2000:
                            
                            sat_xs.append(x_arr[0])   # unwrap the 1-element array, shift to local coords
                            sat_ys.append(y_arr[0])
                            times.append(jd)
                
                        
                    if len(sat_xs) < 2:
                        continue
    
                    sat_xy = np.column_stack([sat_xs, sat_ys])
                    
                    score, mean_dist, spread = self.select_mask(spline_xy, sat_xy)
                    
                    candidates.append((score, sat_name, sat_xy, mean_dist, spread, norad_id))
                    candidates.sort(key=lambda c: c[0])
            
                for score, sat_name, sat_xy, mean_dist, spread, norad_id in candidates:
                    print(f"{sat_name}: score={score:.2f}  mean_dist={mean_dist:.2f}  spread={spread:.2f}")
                    
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (40, 40))
                
                if candidates:
                    best_score, best_name, best_xy, best_mean_dist, best_mean_spread, best_norad_id = candidates[0]
                    ax2.plot(best_xy[:,0], best_xy[:,1], label=f"{best_name} (best match)", linewidth = 3)
                    ax2.scatter(best_xy[:,0], best_xy[:,1], s=6)
                    ax2.set_title(f"Most likely satellite match: {best_name}")
                else:
                    best_norad_id = None
                    ax2.set_title("No satellite match found")
                
                ax1.plot(x_seg, y_seg, color="lime", linewidth=1.5)
                vmin, vmax = np.percentile(self.fits_data, [1, 99.5])
                ax1.imshow(self.fits_data, origin='upper', cmap='gray', vmin=vmin, vmax=vmax)
                ax1.set_xlim(np.min(x_seg) - 100, np.max(x_seg) + 100)
                ax1.set_ylim(np.min(y_seg) - 100, np.max(y_seg) + 100)
                
                ax2.imshow(self.fits_data, origin='upper', cmap='gray', vmin=vmin, vmax=vmax)
                ax2.plot(spline_xs, spline_ys, color = "blue", linewidth = 4, label = "Real Streak")
                ax2.set_xlim(np.min(x_seg) - 100, np.max(x_seg) + 100)
                ax2.set_ylim(np.min(y_seg) - 100, np.max(y_seg) + 100)
                ax2.grid()
                
                ax3.plot(refined["x_refined"], refined["y_refined"], color="lime", label="Refined", linewidth=1.5)

                # x_range = np.linspace(points[:, 0].min(), points[:, 0].max(), 400)
                # y_fit = model.predict(x_range)   

                
                ax3.plot(spline_xs, spline_ys, color = "blue", linewidth = 1)
                ax3.imshow(self.fits_data, origin='upper', cmap='gray', vmin=vmin, vmax=vmax)
                ax3.set_xlim(np.min(x_seg) - 100, np.max(x_seg) + 100)
                ax3.set_ylim(np.min(y_seg) - 100, np.max(y_seg) + 100)
                i+=1
                plt.savefig(f"streak_{i}.png")
                plt.show()
                
                if best_norad_id is not None:
                    ratio = self.computing_ratio_of_lengths(start_point = (spline_xs[0], spline_ys[0]),
                                                    end_point = (spline_xs[-1], spline_ys[-1]), satellite_nid = best_norad_id)
                else:
                    ratio = 1.0
                    print("There was no satellite to compare to!")
                
                mag_arcsec, mag_total, flux_net_list, mag_120_normalised = self.photometry_of_streak(lengths = lengths, 
                                                                                 centres = centres, theta = theta, 
                                                                                 stellar_calib_baseline = mean, 
                                                                                 ratio = ratio, zp_error = std)
                
                
                print(f"Aperture mag/arcmin: {mag_arcsec}")
                print(f"Aperture raw mag: {mag_total}")
                print(f"Aperture 120s normalised mag: {mag_120_normalised}")

    def analyse_simulated_streak(self, results, ra_truth_start, dec_truth_start,
                                  ra_truth_end, dec_truth_end, original_data=None):
        """
        Stripped-down counterpart to plot_streaks() for a synthetic frame with a
        single known-injected streak. No SatChecker involvement anywhere --
        the truth RA/Dec you pass in stands in for the satellite match.
        """
        if original_data is None:
            original_data = self.fits_data
    
        for i, r in enumerate(results):
            refined = self.refine_streak_centerline(
                *r["original_frame_xy"],
                perp_halfwidth=10,
            )
            x_seg, y_seg = refined['x_refined'], refined['y_refined']
    
            cx, cy = self.get_centerline(x_seg, y_seg, n_bins=40)
            tck, u = splprep([cx, cy], s=len(cx) * 10, k=2)
    
            u_check = np.linspace(0, 1, 500)
            spline_xs, spline_ys = splev(u_check, tck)
    
            x_mid, y_mid = splev(0.5, tck)
            _, ra_c, dec_c, _ = xyToRaDecPP(
                [jd2Date(self.exposure_start_jd)], [x_mid], [y_mid], [1], self.pp
            )
    
            # zeropoint calibration -- unchanged, uses field stars only, no satellite involved
            
            # ground-truth-based ratio, replacing the SatChecker ephemeris lookup
            start_point = (spline_xs[0], spline_ys[0])
            end_point = (spline_xs[-1], spline_ys[-1])
            ratio, length_true, length_measured = self.computing_ratio_of_lengths_simulated(
                start_point, end_point,
                ra_truth_start, dec_truth_start, ra_truth_end, dec_truth_end
            )
            print(f"Streak {i}: true length={length_true:.4f} deg, "
                  f"measured length={length_measured:.4f} deg, ratio={ratio:.4f}")
    
            lengths, centres, theta = self.split_streak_into_sections(spline_xs, spline_ys)
    
            lengths, centres, theta = self.split_streak_into_sections(spline_xs, spline_ys)

            result = self.photometry_of_streak_simulated(
                lengths=lengths, centres=centres, theta=theta,
                ratio=ratio, location=(ra_c, dec_c)
            )
            print(result)
    
            # simple 2-panel diagnostic instead of the 3-panel (no satellite-match panel)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            vmin, vmax = np.percentile(self.fits_data, [1, 99.5])
    
            ax1.imshow(self.fits_data, origin='upper', cmap='gray', vmin=vmin, vmax=vmax)
            ax1.plot(x_seg, y_seg, color="lime", linewidth=1.5, label="Hough/refined")
            ax1.set_xlim(np.min(x_seg) - 100, np.max(x_seg) + 100)
            ax1.set_ylim(np.min(y_seg) - 100, np.max(y_seg) + 100)
            ax1.set_title("Refined centerline")
    
            ax2.imshow(self.fits_data, origin='upper', cmap='gray', vmin=vmin, vmax=vmax)
            ax2.plot(spline_xs, spline_ys, color="blue", linewidth=3, label="Fitted spline")
            ax2.set_xlim(np.min(x_seg) - 100, np.max(x_seg) + 100)
            ax2.set_ylim(np.min(y_seg) - 100, np.max(y_seg) + 100)
            ax2.set_title(f"Spline fit vs. injected truth (ratio={ratio:.3f})")
    
            plt.tight_layout()
            plt.savefig(f"streak_simulated_{i}.png")
            plt.show()
