"""SatChecker queries, track matching, length ratio and segment timing."""
import numpy as np
import requests
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.Conversions import jd2Date
from scipy.spatial import cKDTree

# Method bodies are copied unchanged from the original single class.
# They still use self, so they only work as part of OperatingFitsFiles (frame.py).


class SatelliteMixin:
    def streak_passes(self, ra, dec, radius):
        """
        I should really start adding stuff here!!!
        """
        satellites = {}
    
        url = 'https://satchecker.cps.iau.org/fov/satellite-passes/'
        duration = 120
    
        params = {'latitude': self.lat,
                  'longitude': self.lon,
                  'elevation': self.alt,
                  'start_time_jd': self.exposure_start_jd,
                  'duration': duration,
                  'ra': ra,
                  'dec': dec,
                  'fov_radius': radius,
                  'group_by': 'satellite',
                  'async': False}

        # The addition of these guards were implemented with the help of Claude Opus (Sped up the process of writing myself, lazy I know)
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"streak_passes: request failed ({e}) for RA={ra:.4f}, Dec={dec:.4f}")
            return satellites
    
        if r.status_code != 200:
            print(f"streak_passes: HTTP {r.status_code} for RA={ra:.4f}, Dec={dec:.4f} — {r.text[:200]}")
            return satellites
    
        try:
            data = r.json()
        except ValueError:
            print(f"streak_passes: response wasn't valid JSON for RA={ra:.4f}, Dec={dec:.4f}")
            return satellites
    
        if not isinstance(data, dict) or 'data' not in data or 'satellites' not in data.get('data', {}):
            print(f"streak_passes: no satellite data in response for RA={ra:.4f}, Dec={dec:.4f} — {data}")
            return satellites
    
        sat_block = data['data']['satellites']
        if not sat_block:
            print(f"streak_passes: no satellites found within radius={radius} of RA={ra:.4f}, Dec={dec:.4f}")
            return satellites
    
        for sat_key, sat_data in sat_block.items():
    
            if sat_key not in satellites:
                satellites[sat_key] = []
    
            for position in sat_data['positions']:
    
                x, y = self.sky_to_pixel(np.atleast_1d(position['ra']), np.atleast_1d(position['dec']),
                                    position['julian_date'], self.pp)
                satellites[sat_key].append([
                    x,
                    y,
                    position['ra'],
                    position['dec'],
                    position['julian_date'],
                    position['range_km'],
                    sat_data['norad_id']
                ])
    
        return satellites

    def select_mask(self, spline_coords, streak_coords):
        spline_coords = np.asarray(spline_coords)
        satellite_coords = np.asarray(streak_coords)

        if len(spline_coords) < 2:
            return np.inf

        tree_spline = cKDTree(spline_coords)
        tree_satellite = cKDTree(satellite_coords)

            # distance from each satellite point to nearest spline point
        d_satellite_to_spline, _ = tree_spline.query(satellite_coords)
        # distance from each spline point to nearest satellite point
        d_spline_to_satellite, _ = tree_satellite.query(spline_coords)

        all_d = np.concatenate([d_satellite_to_spline, d_spline_to_satellite])

        mean_dist = np.mean(all_d)
        spread = np.std(all_d)
    
        # weight can be tuned - spread penalizes tracks that don't match the streak's shape/length
        score = mean_dist + spread
        return score, mean_dist, spread

    def computing_ratio_of_lengths(self, start_point, end_point, satellite_nid):

        ra0, dec0 = self.pixel_to_sky([start_point[0]], [start_point[1]])
        ra1, dec1 = self.pixel_to_sky([end_point[0]], [end_point[1]])

        S0 = SkyCoord(ra = ra0 * u.deg, dec = dec0 * u.deg)
        S1 = SkyCoord(ra = ra1 * u.deg, dec = dec1 * u.deg)
        exposure_jd = 120 / 86400  # 0.00138889
        
        params = {'catalog': satellite_nid,'latitude': self.lat,'longitude': self.lon, 'elevation': self.alt,'startjd': self.exposure_start_jd,'stopjd': self.exposure_start_jd + exposure_jd,'stepjd': exposure_jd}   # step = full range → only 2 points returned}
        
        r = requests.get('https://satchecker.cps.iau.org/ephemeris/catalog-number-jdstep/', params=params)
        
        data = r.json()['data']
        fields = r.json()['fields']
        p0, p1 = data  # start-of-exposure, end-of-exposure records
        
        ra_i = fields.index('right_ascension_deg')
        dec_i = fields.index('declination_deg')
        
        P0 = SkyCoord(ra=p0[ra_i] * u.deg, dec=p0[dec_i] * u.deg)
        P1 = SkyCoord(ra=p1[ra_i] * u.deg, dec=p1[dec_i] * u.deg)
        length_1 = P0.separation(P1)   # returns an Angle object
        length_2 = S0.separation(S1)
        
        ratio = length_2 / length_1  # 
        return ratio

    def segment_temporal_lengths(self, centres, ratio):
        streak_duration = 120 * np.absolute(ratio)
        # as done before 
        xs = [c[0] for c in centres]
        ys = [c[1] for c in centres]
        ra, dec = self.pixel_to_sky(xs, ys)
        altaz_frame = AltAz(obstime=self.exposure_start, location=self.location)     
                
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        streak_altaz = coords.transform_to(altaz_frame)
        altitude_segment = streak_altaz.alt.deg
        
        segment_arcs = coords[:-1].separation(coords[1:])
        segment_arcs = np.concatenate([[0 * u.deg], segment_arcs])
        cumulative_arcs = np.cumsum(segment_arcs)

        total_measured_arc = cumulative_arcs[-1]
        
        time_at_centres = (cumulative_arcs / total_measured_arc) * streak_duration

        # per-segment dwell time = local time spacing around each point
        dt_segment = np.gradient(time_at_centres.value)  # seconds, same length as segments
    
        return dt_segment, time_at_centres.value, altitude_segment

    def computing_ratio_of_lengths_simulated(self, start_point, end_point,
                                       ra_truth_start, dec_truth_start,
                                       ra_truth_end, dec_truth_end):
        
        ra0, dec0 = self.pixel_to_sky([start_point[0]], [start_point[1]])
        ra1, dec1 = self.pixel_to_sky([end_point[0]], [end_point[1]])
    
        S0 = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg)
        S1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    
        P0 = SkyCoord(ra=ra_truth_start * u.deg, dec=dec_truth_start * u.deg)
        P1 = SkyCoord(ra=ra_truth_end * u.deg, dec=dec_truth_end * u.deg)
    
        length_1 = P0.separation(P1).deg   # -> plain float/array, no unit
        length_2 = S0.separation(S1).deg
    
        length_1 = float(np.atleast_1d(length_1)[0])
        length_2 = float(np.atleast_1d(length_2)[0])
    
        ratio = length_2 / length_1
        return ratio, length_1, length_2
