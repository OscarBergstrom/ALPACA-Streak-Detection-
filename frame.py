"""Loading the FITS frame, cropping and undistorting. Also assembles the full class."""
from datetime import datetime, timedelta, timezone
from . import config

import cv2
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from photutils.centroids import centroid_2dg, centroid_sources
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.Conversions import date2JD
from RMS.Formats.Platepar import Platepar
import logging
logging.basicConfig(level=logging.INFO)
for name in ("matplotlib", "matplotlib.font_manager", "PIL", "astropy", "astroquery", "photutils", "scipy", "skimage", "cv2"):
    logging.getLogger(name).setLevel(logging.WARNING)

from .calibration import CalibrationMixin
from .detection import DetectionMixin
from .photometry import PhotometryMixin
from .pipeline import PipelineMixin
from .satellites import SatelliteMixin

# Method bodies are copied unchanged from the original single class.
# They still use self, so they only work as part of OperatingFitsFiles (frame.py).


class OperatingFitsFiles(DetectionMixin, CalibrationMixin, PhotometryMixin,
                         SatelliteMixin, PipelineMixin):
    """
    I like my classes, they are easy to work with
    """
    def __init__(self, file_name, site, simulated):
        self.file_name = file_name
        self.site = site
        self.median_fwhm  = 10
        self.simulated = simulated

    def opening_fits_file(self):
        with fits.open(self.file_name, memmap=False) as hdul:
            header = hdul[0].header
            self.header = header
            self.fits_data = np.flipud(hdul[0].data)
            # print(header)
            exposure_start = Time(header['DATE-OBS'], scale = 'utc') 
            
            self.exposure_start_jd = exposure_start.jd 
            self.exposure_start = Time(header["DATE-OBS"], format="isot", scale="utc")
            self.exposure_start_dt = datetime.fromisoformat(f"{exposure_start}").replace(tzinfo=timezone.utc)
            self.exposure_time = header.get('EXPTIME', 0)
            self.exposure_end_dt = self.exposure_start_dt + timedelta(seconds = self.exposure_time)
            self.right_ascension = header.get('RA', 0)
            self.declination = header.get('DEC', 0)
            self.ra = header['RA'] # Not sure why I did this twice, but will clear up post project!
            self.dec = header['DEC']
            print("OK", self.ra, self.dec)
            self.lon = header[f'{config.LONGITUDE_HEADER}']
            self.lat = header[f'{config.LATITUDE_HEADER}']
            self.alt = header[f'{config.ALTITUDE_HEADER}']
            self.location = EarthLocation(lat=self.lat*u.deg, lon=self.lon*u.deg, height=self.alt*u.m)

            if self.simulated:
                self.ra_truth_start = header['TRUERA0']
                self.dec_truth_start = header['TRUEDEC0']
                self.ra_truth_end = header['TRUERA1']
                self.dec_truth_end = header['TRUEDEC1']
        
        star_names = "Spica" 

        star_location = SkyCoord.from_name(star_names)
        
        pp = Platepar()
        pp.read(r"C:\Users\carlo\Summer Project\newlenscalib.cal")
        star_x, star_y = raDecToXYPP(np.atleast_1d(star_location.ra.deg), np.atleast_1d(star_location.dec.deg), self.exposure_start_jd, pp)
        
        centroid_func = centroid_2dg
        # print(star_x, star_y)
        x_ref, y_ref = centroid_sources(
                self.fits_data, star_x, star_y, box_size=39, centroid_func=centroid_func
            )

        if np.hypot(star_x - x_ref, star_y - y_ref)[0] < config.CHECK_STAR_TOLERANCE_PX:
            self.pp = pp
        else:
            self.pp = Platepar()
            self.pp.read(str(config.PLATEPAR_FALLBACK))
        print(config.PLATEPAR_PRIMARY, config.PLATEPAR_PRIMARY.exists())
    def sky_to_pixel(self, ra_deg, dec_deg):
        return raDecToXYPP(np.atleast_1d(ra_deg), np.atleast_1d(dec_deg), self.exposure_start_jd, self.pp)

    def pixel_to_sky(self, x, y):
        n = len(np.atleast_1d(x))
        _, ra, dec, _ = xyToRaDecPP([self.exposure_start_jd] * n,
                                    np.atleast_1d(x), np.atleast_1d(y), np.ones(n), self.pp, jd_time = True)
        return ra, dec
        # print(self.pp)
    
    @staticmethod
    def inverse_gnomonic(xi, eta, ra_centre_deg, dec_centre_deg):
        """
        Inverse gnomonic projection. Valid for xi, eta < pi/2 radians!
        """
        # Converts degress -> radians
        ra0 = np.radians(ra_centre_deg) 
        dec0 = np.radians(dec_centre_deg)
        
        rho = np.sqrt(xi**2 + eta**2)
        c = np.arctan(rho)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        with np.errstate(invalid="ignore", divide = "ignore"):
            dec = np.where(
                rho > 1e-12, # Stops divide by 0 errors! Helped via Claude!
                np.arcsin(cos_c * np.sin(dec0) + (eta * sin_c * np.cos(dec0)) / rho),
                dec0,
            )
            ra = np.where(
                rho > 1e-12,
                ra0 + np.arctan2(
                    xi * sin_c,
                    rho * np.cos(dec0) * cos_c - eta * np.sin(dec0) * sin_c,
                ),
                ra0,
            )
            return np.degrees(ra) % 360.0, np.degrees(dec)

    def get_obs_time_tuple(self):
        dt = self.exposure_start.datetime
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond // 1000)
    
    @staticmethod
    def great_circle_destination(ra0_deg, dec0_deg, bearing_deg, ang_dist_deg):
        ra0 = np.radians(ra0_deg)
        dec0 = np.radians(dec0_deg)
        brng = np.radians(bearing_deg)
        d = np.radians(ang_dist_deg)
    
        dec = np.arcsin(np.sin(dec0) * np.cos(d) + np.cos(dec0) * np.sin(d) * np.cos(brng))
        ra = ra0 + np.arctan2(
            np.sin(brng) * np.sin(d) * np.cos(dec0),
            np.cos(d) - np.sin(dec0) * np.sin(dec),
        )
        return np.degrees(ra) % 360.0, np.degrees(dec)

    def cutoff_boundary_pixels(self, ra_centre_deg, dec_centre_deg, cutoff_deg, n_points=720):
        bearings = np.linspace(0, 360, n_points, endpoint=False)
        ra_line, dec_line = self.great_circle_destination(ra_centre_deg, dec_centre_deg, bearings, cutoff_deg)

        x_pix, y_pix = self.sky_to_pixel(ra_line, dec_line)
        return x_pix, y_pix

    def crop_fits_by_angle(self, fits_out_path, cutoff_deg=config.CUTOFF_DEG,
                            preview_png="crop_preview.png"):
        """
        Crops the raw fits exposure to eliminate buildings and ensure that gnomonic map doesnt explode at 90degrees.
        """
        
        height, width = self.fits_data.shape[-2], self.fits_data.shape[-1]
        ra_centre_deg, dec_centre_deg = self.ra, self.dec
    
        x_bound, y_bound = self.cutoff_boundary_pixels(ra_centre_deg, dec_centre_deg, cutoff_deg)
        
        # --- build mask from the boundary polygon ---
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = np.stack([x_bound, y_bound], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 1)
    
        # --- zero out everything outside the mask ---
        masked_data = self.fits_data.copy()
        masked_data[mask == 0] = 0
    
        # --- crop to the mask's bounding box ---
        ys, xs = np.where(mask == 1)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        cropped = masked_data[y0:y1 + 1, x0:x1 + 1]
    
        print(f"Original size: {width} x {height}")
        print(f"Cropped size:  {x1 - x0 + 1} x {y1 - y0 + 1}")
        print(f"Crop offset (add back before using the platepar again): X0={x0}, Y0={y0}")
        header = self.header.copy()
        # --- record the crop offset in the header for later use ---
        header["HISTORY"] = f"Cropped to {cutoff_deg} deg from boresight (RA={ra_centre_deg:.4f}, Dec={dec_centre_deg:.4f})"
        header["CROP_X0"] = (x0, "X offset of this crop within the original frame")
        header["CROP_Y0"] = (y0, "Y offset of this crop within the original frame")
        header["CROP_ANG"] = (cutoff_deg, "Angular radius (deg) used for this crop")

        self.x_offset = x0
        self.y_offset = y0
        self.cropped_data = cropped # will adjust ofc!
        self.cutoff_deg = cutoff_deg
        
        fits.writeto(fits_out_path, cropped, header, overwrite=True)
        print(f"Wrote cropped FITS: {fits_out_path}")
    
        return fits_out_path

    def run_crop_procedure(self):
        self.crop_fits_by_angle(
            fits_out_path=f"{self.file_name}_cropped{config.CUTOFF_DEG}deg.fits",
            cutoff_deg=config.CUTOFF_DEG,  # pick anywhere in your 65-70 deg range
        )

    def undistorted_xy_to_original_xy(self, px, py, out_width, out_height, half_tan,
                                       ra_centre_deg, dec_centre_deg, obs_time_tuple,
                                       crop_x0=0, crop_y0=0):
        """
    
        """
        px = np.atleast_1d(np.asarray(px, dtype=np.float64))
        py = np.atleast_1d(np.asarray(py, dtype=np.float64))
    
        xi = (px / (out_width - 1) - 0.5) * 2 * half_tan
        eta = (py / (out_height - 1) - 0.5) * 2 * half_tan
    
        ra_deg, dec_deg = self.inverse_gnomonic(xi, eta, ra_centre_deg, dec_centre_deg)
        
        x_orig, y_orig = self.sky_to_pixel(ra_deg, dec_deg)
    
        # x_orig, y_orig are in the ORIGINAL (uncropped) frame's coordinates.
        # Convert to the CROPPED frame's coordinates (what we actually remap from).
        return x_orig - crop_x0, y_orig - crop_y0

    def build_dense_undistortion_map(self, out_width, out_height, half_tan, ra_centre_deg, dec_centre_deg,
                                      pp, obs_time_tuple, crop_x0, crop_y0):
        yy, xx = np.mgrid[0:out_height, 0:out_width]
        x_crop, y_crop = self.undistorted_xy_to_original_xy(
            xx.ravel(), yy.ravel(), out_width, out_height, half_tan,
            ra_centre_deg, dec_centre_deg, obs_time_tuple, crop_x0, crop_y0)
        map_x = x_crop.reshape(out_height, out_width).astype(np.float32)
        map_y = y_crop.reshape(out_height, out_width).astype(np.float32)
        return map_x, map_y

    def undistort_cropped_fits(self, cropped_fits_path, fits_out_path,
                                out_width=None, out_height=None):
        
        with fits.open(cropped_fits_path, memmap=False) as hdul:
            header = hdul[0].header.copy()
    
        ra_centre_deg, dec_centre_deg = self.ra, self.dec
        obs_time_tuple = self.get_obs_time_tuple()
    
        half_tan = np.tan(np.radians(self.cutoff_deg))
    
        if out_width is None:
            out_width = self.cropped_data.shape[-1]
        if out_height is None:
            out_height = self.cropped_data.shape[-2]
    
        print(f"Building dense analytic undistortion map: {out_width} x {out_height} pixels...")
        map_x, map_y = self.build_dense_undistortion_map(
            out_width, out_height, half_tan, ra_centre_deg, dec_centre_deg,
            self.pp, obs_time_tuple, self.x_offset, self.y_offset,
        )
    
        corrected = cv2.remap(self.cropped_data.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_CUBIC)
        self.corrected_data = corrected.copy()
        # Save everything needed to invert this later (streak -> distorted frame)
        header["HISTORY"] = "Dense analytic undistortion (tangent-plane / gnomonic)"
        header["UND_RA0"] = (ra_centre_deg, "Tangent point RA (deg) used for undistortion")
        header["UND_DEC0"] = (dec_centre_deg, "Tangent point Dec (deg) used for undistortion")
        header["UND_HTAN"] = (half_tan, "Half-width of tangent-plane extent (tan(cutoff_deg))")
        header["UND_W"] = (out_width, "Undistorted image width used to build this map")
        header["UND_H"] = (out_height, "Undistorted image height used to build this map")
        # CROP_X0/CROP_Y0 already present from the crop step -- keep them; they're
        # still needed to map all the way back to the ORIGINAL uncropped frame.
    
        fits.writeto(fits_out_path, corrected, header, overwrite=True)
        print(f"Wrote undistorted FITS: {fits_out_path}")
        return fits_out_path

    def undistort_fits(self):
        self.undistort_cropped_fits(
            cropped_fits_path=f"{self.file_name}_cropped{self.cutoff_deg}deg.fits",
            fits_out_path=f"{self.file_name}_undistorted.fits",
        )
