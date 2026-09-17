from alpaca.frame import OperatingFitsFiles

frame = OperatingFitsFiles("path/to/frame.fits", site="paranal", simulated=False)
frame.opening_fits_file()
frame.run_crop_procedure()
frame.undistort_fits()
frame.run_hough_transform()
