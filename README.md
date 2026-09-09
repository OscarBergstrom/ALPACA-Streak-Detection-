# ALPACA-Streak-Detection-
This repository represents my work during my MSc project under Dr Michael Peel at Imperial College London.

This pipeline currently takes in ALPACA (ALL-sky Paranal Apical CAmera) downloaded from the ESO archive:
https://archive.eso.org/eso/eso_archive_main.html.

This camera records data using a distorted fisheye lens, with a monochromatic (400-700nm) 8750 × 8750-pixel output. Because the ALPACA lens has no solved WCS, we applied an astrometric calibration method using SkyFit2 [1][2]. SkyFit2 applies a distortion polynomial to solve for the fisheye lens calibrated on known stars; it also provides functions to convert raw pixel space (x,y) into celestial coordinates (RA/DEC) and vice versa.

The pipeline works as follows: 
- Images are cropped to remove buildings and high-extinction, low-altitude viewing.
- An inverse Gnomonic Projection is applied. In this process, the Gnomonic projection is first normalised to the initial FITS frame, then for each pixel in gnomonic space (x', y') this pixel is projected into its celestial space equivalent (RA/DEC). This (RA/DEC) space is then converted back to the original pixel space using SkyFit2.
- A set of preprocessing steps in line with methods used by the SatMetrics pipeline [3]; this includes brightness thresholds, binarisation, and a Canny filter.
- From this preprocessed image, Hough transforms and line-fitted contours to extract glints.
- A kd-tree is then used to compare SatChecker API FOV passes data [4].
- Next, photometry is performed using rectangular apertures; current methods can include Curve-of-Growth and an analytic 1D Moffat fit.
- Local Reference stars are used to perform relative photometry, with an equal Enclosed Energy (EE) ratio to that of the satellite streak.
- Magnitudes are dwell time corrected according to the TLE data.  


This is still a work in progress, and now, with some extra time after submission, I will clean up the code to streamline and improve the pipeline. 


References:
[1] https://globalmeteornetwork.org/wiki/index.php?title=SkyFit2
[2] Denis Vida, Damir Šegon, Peter S Gural, Peter G Brown, Mark J M McIntyre, Tammo Jan Dijkema, Lovro Pavletić, Patrik Kukić, Michael J Mazur, Peter Eschman, Paul Roggemans, Aleksandar Merlak, Dario Zubović, The Global Meteor Network – Methodology and first results, Monthly Notices of the Royal Astronomical Society, Volume 506, Issue 4, October 2021, Pages 5046–5074
[3] https://github.com/uwescience/satmetrics SatMetrics, University of Washington.
[4] https://satchecker.readthedocs.io/en/stable/fov.html
