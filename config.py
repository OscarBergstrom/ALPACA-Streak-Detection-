""""Configuration file for the pipeline."""
from pathlib import Path

# SUPER CONTROLS
WRITE_INTERMEDIATES = True # Allows the user to save the intermediate files (cropped, undistorted, etc.) to disk for inspection

# Paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = PROJECT_ROOT / "calib_files"
PLATEPAR_PRIMARY = CALIB_DIR / "newlenscalib.cal"       # frame.py: opening_fits_file
PLATEPAR_FALLBACK = CALIB_DIR / "lenscalibMAIN.cal"     # frame.py: opening_fits_file
STELLAR_CATALOGUE = PROJECT_ROOT / "stellar_calibration1.txt"  # calibration.py: stellar_calibrations

# Exposure processing parameters ** THIS WILL NEED TO BE CHANGED FOR DIFFERENT LENSES **
LONGITUDE_HEADER = "HIERARCH ESO TEL GEOLON"
LATITUDE_HEADER = "HIERARCH ESO TEL GEOLAT"
ALTITUDE_HEADER = "HIERARCH ESO TEL ALT"
CUTOFF_DEG = 60.0
## For FITS headers with no location information, the following coordinates may be used for the pointing    
SITE = "PARANAL"

# Platepar processing parameters 
CHECK_STAR = "Spica"
CHECK_STAR_TOLERANCE_PX = 10
CENTROID_BOX_PX = 39

# Optics 
OPTICAL_CENTRE_PX = (4375, 4375)
VIGNETTING_COEFF = 0.000204
VIGNETTING_COEFF_ERR = 0.000015
VIGNETTING_SAFE_RADIUS_PX = 3500
PLATE_SCALE_ARCMIN_PER_PX = 1.17   

# Detection parameters


HOUGH_KWARGS = dict(rho=1, theta_deg=0.5, hough_threshold=50,
                    min_line_length=200, max_line_gap=10)
MERGE_KWARGS = dict(dist_tol=40, angle_tol_deg=5)
MIN_STREAK_LENGTH_PX = 200


# UNDISTORTED_CENTRE_PX = (3246, 3246)
# MAX_STREAK_DISTANCE_PX = 3000

#  Photometry parameters
ENCLOSED_ENERGY = 0.80 