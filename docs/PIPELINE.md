HOW THE RETRIEVAL WORKS
=======================
Two config files decide everything: retrieval.yaml (every threshold and
method choice, shared by every site so results stay comparable) and
site_<radar>.yaml (paths, field names, freezing-level sources, run policy).
windvel.config.load_config validates both once, refuses unknown keys, and
hands the stages one resolved mapping; every key named below is required.
The numbers quoted are the ones examples/retrieval.yaml ships; where a key
is named without a number, read the value from the file.

Per input file the RHI sweeps are found (extract_rhi_sweep_indices), then:

    apply_corrections            stage 0
    select_cloud_transects       stage 1
    calculate_windvel            stages 1 (selection) to 7
    save_windvel_file, plot_windvel_summary


0. CORRECTIONS                                            corrections:
   apply_corrections()

   Runs before anything reads the input fields. Each enabled correction is
   written as a NEW field; the delivered field is never modified and both
   are saved. Every later consumer of that variable -- the cloud contour,
   the weak-echo anchor screen, the fall speed -- reads the corrected field
   while the correction is on.

   attenuation (correct: false). At C and X band the beam loses power
   passing through rain, so a gate behind a cell reads low. The fall-speed
   relations were built at S band where that loss is negligible, so an
   attenuated Z biases Vsed low and, since dw/dVsed = 1, w with it. The
   specific attenuation Ah = a * Z^b is accumulated along the ray and
   doubled, exclusive of the gate's own contribution. It refuses to guess:
   a and b are band specific, so `relation` names an entry of
   windvel.tables.ATTENUATION_RELATIONS or gives {band, a, b} inline, and
   the radar's own frequency is checked against the relation's band
   (require_band_match). It runs below the melting level only (liquid_only)
   and the two-way total is capped (max_correction_db): the estimate reads
   attenuation from an already-attenuated Z, so it under-corrects and the
   shortfall compounds along the ray. Gates at the cap are counted.


1. CLOUD OBJECTS                        cloud_objects:, object_selection:
   select_cloud_transects(), apply_object_selection_criteria()

   Per sweep, contiguous echo at or above reflectivity_threshold_dbz (-20)
   is contoured and labelled into local_object_ids, largest first. A
   contour whose true area -- from the gate geometry, not a pixel count:
   one degree spans 87 m at 5 km and 520 m at 30 km -- is under
   min_area_km2 (7) is dropped. gate_quality lists per-gate screens
   ({field, min}) applied first; the config decides, never the file, so a
   named field missing from a file is an error and [] means no screen.
   object_boundaries marks a shell boundary_thickness_gates wide around
   each object, leaving the lowest exclude_lowest_rays rays of each sweep
   unmarked because the beam grazes the ground there.

   Which clouds are worth retrieving is decided ONCE, here: an object whose
   horizontal extent is outside min_span_km..max_span_km, whose top is
   below min_top_km, or which is thinner than min_depth_km has its id
   zeroed, and no later stage computes for it. Selection limits the SIZE of
   an object, never its range from the radar. Every rejection is counted
   and logged by criterion.


2. SEDIMENTATION -- how fast are the particles falling?   sedimentation:
   get_sed_vel()

   The Doppler velocity contains the particles' own fall speed, which has
   to be removed before anything else means anything. Three methods:

     pid_giangrande_oklahoma2013   needs a particle id (input_variables.pid_var).
                                   Rain by a power law in Z, graupel by an
                                   offset square root, ice held at 2 m/s.
     pid_giangrande_darwin2026     a single power law per class, ice included.
     reflectivity_temperature      no particle id: branches on reflectivity and
                                   the SIGN of the temperature only. The freezing
                                   level comes from the site's environment
                                   ladder -- a temperature field in the file,
                                   else the nearest sounding in sounding_dir
                                   (bounded in age and distance), else
                                   temperature_profile, else melting_level_km --
                                   and the retrieval RAISES rather than assume
                                   one.

   All three are scaled for air density, Vt(z) = Vt0 * exp(z / 8.5 km)^0.4.
   The output field's attributes record the method, the reflectivity field
   read, and where the freezing level came from.

   Why it matters: dw/dVsed = 1 exactly, so an error here is an equal and
   opposite error in w. The methods differ by up to about 2 m/s in ice.


3. HORIZONTAL WIND -- the hard part                     horizontal_wind:
   get_horizontal_velocity()

   In an RHI the radial velocity mixes the horizontal and vertical wind.
   The trick is that at low elevation the beam is nearly horizontal, so
   gates there see mostly horizontal wind. The candidate wind at every
   gate is

       hv_candidate = (Vr + Vsed*sin e) / cos e

   which equals the true horizontal wind only where the air is not moving
   vertically -- in weak echo at the cloud's edges. Each object (one
   contour in ONE sweep) is cut into bin_size_m (300 m) altitude bins from
   its own base. In each bin the gates are split at the median slant range
   into a NEAR half and a FAR half, and each half yields one anchor by the
   first rung of the ladder it can satisfy. The code in front of each rung
   is the value horizontal_velocity_source carries:

     1  MEASURED      the median (anchor.robust) of the half's WEAK-ECHO
                      gates: reflectivity below the first entry of
                      anchor.weak_echo_dbz ([25, 30]), falling back to the
                      second entry where the strict one cannot muster
                      anchor.min_gates of them. Weak echo because that is
                      where the fall speed contaminates the estimate least.
                      Gates steeper than anchor.max_elevation_deg
                      (hv = u + w tan e: a steep beam cannot separate the
                      two) or implying more than anchor.max_candidate_mps
                      do not vote.
     2  MIRRORED      NEAR edge only (fill_mirror): the median of the far
                      half's innermost min_gates gates becomes the near
                      anchor, and the far anchor is rebuilt from the gates
                      left over, so both ends of the bin are real gates at
                      opposite ends of the measured span. Needs the far
                      half to hold 2 * min_gates measured gates. An empty
                      FAR edge is never filled from the near side.
     6  INTERPOLATED  a run of at most fill_interpolate_max_gap_bins empty
                      bins with a finite anchor above AND below is filled
                      linearly in altitude, each edge on its own -- never at
                      a profile's ends, where a one-sided fill would be
                      extrapolation.
     0  NONE          the bin stays empty and produces no output.

   The assembled edge profiles then pass through ONE spike correction,
   smooth_edge_profile(), once per entry of spike_correction.passes
   (half-windows of +-3 bins, then +-5). A point is flagged when it
   departs from a line fitted through its surviving neighbours by more
   than max(shear_per_km * dz, k_mad robust sigmas of the profile's own
   residuals) -- worst first, refitting after each removal -- and is
   replaced by interpolation between the nearest survivors. At most
   max_fraction of a profile is rewritten and the end points are left as
   retrieved. A rewritten bin reports 5 SMOOTHED. Bins thinner than
   min_bin_gates are not retrieved (0 = off).

   The wind across a bin is a straight line in slant range between its two
   anchors. horizontal_velocity_source records, per gate, the WEAKEST rung
   of the two edges it sits between, with SMOOTHED outranking all: whatever
   rung produced the value, a fit has since replaced it. The settings the
   retrieval used are stamped into the field's attributes, and the per-code
   gate tally goes to the log.


4. VERTICAL VELOCITY                           vertical_velocity.compute:
   calculate_vertical_velocity()

       w = (Vr + Vsed*sin e - Vh*cos e) / sin e

   for every object gate within max_horizontal_distance_km of the radar --
   a per-gate cut, so a near gate is never denied a w because its cloud's
   far end lies beyond the limit -- on rays above min_elevation_deg, where
   1/sin(e) still means something. Every error is amplified toward the
   horizon, 2x at 30 degrees and 11x at 5. Values are kept EVERYWHERE they
   can be computed, which is why the field contains 50 m/s "updrafts" at
   low elevation. They are flagged, not deleted -- see stage 6.


5. ERROR                                         vertical_velocity.error:
   calculate_vertical_velocity_error()

       dW = sqrt[ (dVr/sin e)^2 + dVsed^2 + (dVh1^2 + dVh2^2) * cot^2 e ]

   dVr (0.2), dVsed (2) and dVh2 (2) are scalars; dVh1, the anchor error,
   is tabulated against elevation (dVh1_table: a name from
   windvel.tables.VERTICAL_ERROR_TABLES, or an inline table). A function of
   ELEVATION ONLY -- a measured anchor and a mirrored one report the same
   sigma -- so the error field is a property of the viewing geometry,
   comparable between fill methods and between campaigns; the provenance is
   published separately for anyone who wants to combine the two. With the
   shipped table: about 2 m/s at zenith, 6-7 at 30 degrees, 21 at 10 and
   over 40 at 5 -- beyond any plausible w, which is the quantitative form
   of the argument for restricting analysis to high elevation. The floor is
   the sedimentation error: you cannot know w better than you know the
   fall speed, whatever the geometry.


6. USABILITY                                    vertical_velocity.usable:
   vertical_velocity_usability()

   vertical_velocity_flag is 1 where the elevation exceeds
   min_elevation_deg (30), the horizontal distance is within
   max_horizontal_distance_km (null = off; beam broadening is then
   unrepresented anywhere in the output) and |w| is below max_abs_w_mps
   (60: a retrieval that ran on bad inputs, not a measurement). A FLAG,
   never a filter: masking would conflate 'outside the trusted domain'
   with 'no retrieval possible', and the values would be lost on write.


7. COHERENT OBJECTS                                     coherent_objects:
   detect_coherent_objects(), coherent_structures.py

   Two definitions, both computed, both labelled per sweep on the UNMASKED
   w so no structure is clipped by a boundary that has nothing to do with
   it:

     threshold    connected |w| above threshold.w_mps (6), kept if its AREA
                  reaches threshold.min_area_km2 (0.1).
     persistence  merge-tree segmentation of w/sigma. Sweeping down from
                  the highest value, each local maximum starts a component;
                  where two meet is a saddle and the weaker dies there. A
                  core survives if it stands persistence.min_sigma (2) above
                  that saddle, and is outlined at max(saddle,
                  persistence.floor_sigma (1)). Working in units of sigma is
                  the point: 8 m/s is a real updraft looking straight up and
                  noise at 5 degrees elevation, and one number cannot mean
                  both. There is no speed threshold, so a persistence
                  object can be weak in m/s.

   Both are 8-connected. Two rules keep the objects honest. An object is
   drawn as the piece of its region that actually contains its peak, since
   a label can survive where the gates linking it to the core fell below
   the cut. And a core lying WITHIN another is absorbed into it: a core
   beside another is a boundary between two structures, a core inside
   another is a dip within one.

   Each object then gets two INDEPENDENT verdicts (coherent_objects.usable):

     _flag        usable: at least min_area_fraction (0.7) of its area
                  inside the trusted domain AND the majority of its core
                  inside. The core is the connected patch of strongest gates
                  around the peak, started at core_peak_fraction of the peak
                  and grown until it holds core_min_gates, located by
                  |w|/sigma rather than by speed -- an object detected on
                  significance must have its core found the same way.
     _truncated   runs off the edge of what was sampled, as a bitmask of
                  which edge (TRUNC_TOP, TRUNC_BOT, TRUNC_NEAR, TRUNC_FAR)
                  or abuts an interior hole where the retrieval produced
                  nothing (TRUNC_GAP). Such objects are real but
                  INCOMPLETELY OBSERVED: their area, depth and peak are
                  lower bounds. Kept separate from usable because they
                  answer different questions -- count truncated objects
                  when counting updrafts, exclude them when averaging size.


FIELDS WRITTEN                          calculate_windvel.OUTPUT_FIELDS
    reflectivity_corrected              only while a correction is on
    local_object_ids                    cloud objects per sweep; 0 = rejected
    object_boundaries
    sedimentation_velocity              attributes: method, field, freezing level
    horizontal_velocity
    horizontal_velocity_source          which rung produced each gate
    vertical_velocity                   values kept everywhere, never masked
    vertical_velocity_error             one sigma, from the geometry
    vertical_velocity_flag              1 where trusted
    coherent_objects_threshold          signed labels, + up, - down
    coherent_objects_threshold_flag
    coherent_objects_threshold_truncated
    coherent_objects_persistence
    coherent_objects_persistence_flag
    coherent_objects_persistence_truncated

Every output file also carries windvel_version, the retrieval file and its
version, any overrides, and the site file, so a result can always be traced
to the code and the numbers that made it. Every skip, cap or drop on the
way is counted and written to the run log.


