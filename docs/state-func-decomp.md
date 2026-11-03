# Task: Global-mean state, function and interaction decomposition of area-mean ET, relative to observations

Follow CLAUDE.md throughout. This is an analysis outside the pipeline, like
`regress_lai_aridity.py`: do not add it to `scripts/run_all.sh`. Do not change `binning.bin_stats`,
`cfg.FAMILIES`, edge files or any existing pipeline script. Build every path with
`cfg.output_dir`, `cfg.bins_tag`, `cfg.inputs_path`, `cfg.area_mask_path` and `edges.edges_path`,
never by hand.

Every comparison uses an observational reference. Output only scalar (domain-mean) quantities.
The LAI–AI bins are used only to condition the sums; do not save or plot per-bin contributions.

## Design requirement: obs state uncertainty must be addable without code changes
I will later rerun prep_obs.py with several LAI, P and Rn products (not only several ET
products). That changes the obs sources, the pooled_all edges and the common area mask. The
script must handle any set of obs product combinations by reading them from the data:
- Never hard-code obs product names, the number of obs sources, or the assumption that obs
  sources share LAI, P or Rn (i.e. share p).
- Identify each obs source by its product coords `<v>_product` (v in prep_obs.FACTORS; confirm in
  prerequisite 4). Carry these coords onto every output dimension that indexes obs sources.
- Product axes are a module constant, OBS_PRODUCT_AXES = ("et", "lai", "pr", "rn"). Analyses
  grouped by product axis loop over this tuple and skip, with a printed note, any axis that has
  only one product.
- Record the full list of obs sources and their product coords in the attrs of every output.

## Families
- Reference: obs. Each obs source is one ET × LAI × P × Rn product combination (prep_obs.py).
- Evaluated: era5, cmip, fppe, ippe, lens (cfg.FAMILIES minus obs), plus TRENDY:
  - module constant TRENDY_FAMILIES = ("trendy_s3", "trendy_s2")
  - offline DGVMs, one member per model, driven by the same CRU-JRA P and ERA5 Rn
  - AI is therefore identical across TRENDY models (within-TRENDY state differences come only
    from LAI), and prescribed (Rn not from the models' own energy balance). State both in the
    docstring.
  - trendy_s3 is required. trendy_s2 is optional: skip it with a printed note if absent.
- The obs sources are also evaluated against the obs pooled reference (the observational spread).

## Inputs
Land-area-weighted climatological step 14 outputs with the `pooled_all` edges
(`bin_all_clim.py` with ETUNC_AREA_WEIGHTED=1, ETUNC_BIN_VARIABLE=et), per-member files:
  cfg.output_dir(PROC_ROOT, <family>, "qbin", <period>, <bins>) /
    <family>.et_clim.area_weighted.qbin_pooled_all.<bins>.<grid>.<period>.nc
Dims (member, stats, y_bin, x_bin), with `stats` = binning.STATS.

- Do not use:
  - the unweighted `<family>.et_clim.qbin_*` files (no `.area_weighted` tag; these are what
    notebooks/for-abby.ipynb reads)
  - the annual `bin_all.py` outputs (sum_w = area × number of valid years)
  - `lens…members_pooled.nc`
  - `.sources_pooled.nc` (its sum_w is summed over members; pool from the per-member files as
    defined below)
- Edges: `pooled_all` only, so every dataset and the obs reference share one LAI–AI partition.
  Partition sensitivity is tested by bin count, not by edge kind.
- Pin AREA_WEIGHTED = True, BIN_VARIABLE = "et" and EDGES = "pooled_all" as module constants,
  independent of cfg, with a comment explaining why: sum_b p f is the area-mean ET only for
  area-weighted ET, and only a shared partition makes datasets comparable.
- Bin counts come from cfg.N_YBINS / cfg.N_XBINS (ETUNC_* overridable), one combination per run.

## Notation
- m = the dataset being evaluated; r = the observational reference (one obs source, or the obs
  pooled reference).
- i = gridcell; b = LAI–AI bin.
- Every difference is "m minus r".

## Definitions
Domain D = the gridcells i of the common area mask (cfg.area_mask_path), identical for every
dataset.
- a_i = land area of gridcell i (rg.land_area, km2); A = sum_{i in D} a_i = domain land area (km2).
  Both depend only on the grid.
- For dataset m, each gridcell has its own climatological values over cfg.TIME_SLICE:
  - LAI_{m,i} (m2 m-2)
  - AI_{m,i} = Rn_{m,i} / (L_v P_{m,i}) (dimensionless, from m's own Rn and P)
  - e_{m,i} = ET (W m-2)
- B_m(b) = the gridcells whose (LAI_{m,i}, AI_{m,i}) fall in bin b of the shared pooled_all
  partition. The partition is shared; the assignment of gridcells to bins is dataset-specific.
- p_m(b) = sum_{i in B_m(b)} a_i / A
    occupancy: fraction of the domain's land area that m places in LAI–AI regime b
    (dimensionless, sum_b p_m = 1). Equal to `sum_w`(b) / sum_b `sum_w` when every gridcell in
    the mask is binned (prerequisite 2). Area-based, not a gridcell fraction: never use `count`
    or `n_eff`.
- f_m(b) = sum_{i in B_m(b)} a_i e_{m,i} / sum_{i in B_m(b)} a_i
    bin-mean ET: area-weighted mean ET of the land m places in regime b (W m-2; the `mean`
    statistic; NaN if B_m(b) is empty).
- E_m = sum_b p_m(b) f_m(b) = sum_{i in D} a_i e_{m,i} / A
    area-weighted domain-mean ET (W m-2).
- Pooled datasets (multi-member CMIP6 sources, obs_pooled, <family>_pooled), sources s weighted
  once:
    p_pool(b) = mean_s p_s(b)
    f_pool(b) = sum_s p_s(b) f_s(b) / sum_s p_s(b)
  so E_pool = mean_s E_s. For a multi-member source, apply the same formulas over its members
  first. Do NOT use binning.pool_members across sources.

## Prerequisites (check, report, then STOP for my confirmation)
0. TRENDY provenance. Committed cfg.FAMILIES excludes TRENDY, but
   trendy_s3.et_clim.qbin_pooled_all.lai15_ai15.1deg.<period>.nc appears to exist. Determine and
   report:
   a. `git status` / `git diff` of the working tree here: is there an uncommitted change to
      cfg.FAMILIES or BIN_FAMILIES?
   b. Did trendy_s3 contribute to the current common area mask and pooled_all edges, or was it
      only binned onto them? Check the logs of make_edges.py and bin_all_clim.py in
      WORK_ROOT/logs (summarize_settings and git commit), and the mtimes of the mask, the edges
      and the trendy_s3 qbin files.
   c. Were all families' area-weighted files binned with the same mask and edges? Compare the
      edges stored in each file's attrs.
   If (b) or (c) is inconsistent across families, stop and tell me what to rerun. Do not edit
   cfg.FAMILIES, edges or bin scripts.
1. For each family (including both TRENDY families) and bin count under qbin/<period>/: list
   which area-weighted et_clim pooled_all files exist and whether they contain `sum_w`. Tell me
   which runs are missing. Do not launch them.
2. Domain check: for every member, sum_b sum_w equals A to within 1e-10 relative error. Report any
   member with gridcells in the mask missing LAI, AI or ET, or lying outside the outer pooled_all
   edges (most likely for datasets that did not contribute to the edges, e.g. trendy_s2), and
   the area it loses.
3. Identity check: for at least one member per family, E_m = sum_b p_m f_m equals
   sum_i a_i e_{m,i} / A, computed directly from cfg.inputs_path (clim ET as in bin_all_clim.py),
   to within 1e-10 relative error.
4. Member → source mapping (source = `source` coord on `member`, set by the prep scripts):
   - obs, era5, lens, fppe, ippe, trendy_*: every member is its own source.
   - cmip: source = model (source_id), member = "<source_id>.<member_id>". With
     DEFAULT_MEMBERS = "top" this is one member per model; flag any model with more.
   - fppe/ippe: member 0 is the default (lc.FHIST_PPE_MEMBERS); parameter names come from
     lc.ppe_member_name.
   - obs: list the sources, confirm the product coord names, and for each axis in
     OBS_PRODUCT_AXES list its products and whether the combinations form a full factorial
     (prep_obs.py drops combinations with fewer than MIN_YEARS shared years).
5. Obs occupancy: report how much p differs across obs sources, overall and by product axis
   (max over bins of |p_s − p_s'| for pairs that differ only in that axis). Report it rather
   than assuming zero.

## Decomposition (scalar quantities only)
For each evaluated dataset m and obs reference r, on the common support S (bins where f_m and
f_r are both finite), compute four counterfactual domain-mean ET values (W m-2):
  E_rr = sum_{b in S} p_r f_r     observed state, observed function
  E_rm = sum_{b in S} p_r f_m     observed state, m's function
  E_mr = sum_{b in S} p_m f_r     m's state, observed function
  E_mm = sum_{b in S} p_m f_m     m's state, m's function
Physical reading:
  E_rm = the ET m's representation of ET would give if m's climate and vegetation state were
         observed.
  E_mr = the ET the observed LAI–AI–ET relationship would give on m's simulated land state.

The three decomposition terms (W m-2):
  FUNCTION    = E_rm − E_rr     (ET difference from m's ET response, on the observed state)
  STATE       = E_mr − E_rr     (ET difference from m's land state, under the observed response)
  INTERACTION = E_mm − E_rm − E_mr + E_rr
                (> 0: m's state and function biases reinforce; < 0: they compensate)
  FUNCTION + STATE + INTERACTION = E_mm − E_rr

Also store:
- the full-domain means E_m and E_r
- OFF_SUPPORT_m = sum_{b not in S} p_m(b) and OFF_SUPPORT_r = sum_{b not in S} p_r(b)
The off-support ET contributions are E_m − E_mm and E_r − E_rr (derived, not stored).

Obs pooling weights: one weight per obs source (module constant OBS_POOL_WEIGHTING = "source").
- If the obs combinations are not a full factorial, equal source weights give products unequal
  weight. Report the effective weight of each product on each axis in the attrs and the log.
- Keep the weighting in one function, so a hierarchical option (equal weight per product on each
  axis, analogous to edges.hierarchical_weights) can be added later without touching the
  comparisons. Do not implement that option now.

## Comparisons (all with an obs reference)
1. Every evaluated dataset vs. every obs reference: each member of era5, cmip, fppe, ippe, lens,
   trendy_s3 (and trendy_s2 if present), plus each "<family>_pooled", vs. each obs source and
   vs. the obs pooled reference.
2. Observational spread: each obs source vs. the obs pooled reference, with all quantities.
   STATE and INTERACTION measure observed-state uncertainty; FUNCTION measures ET-response
   uncertainty. For each evaluated dataset (vs. obs_pooled), flag any term that lies within the
   obs-source range of that term.
3. Obs-reference sensitivity by product axis: for each evaluated dataset and each quantity, the
   range (max − min) across obs references, overall and grouped by product axis. Within each
   axis, take the range across that axis's products with the other axes held fixed, then the
   mean over the fixed combinations. Skip axes with one product.
4. Variance partition across the sources within each family (obs, cmip, trendy_s3, fppe, ippe,
   lens), all against the obs pooled reference:
     Var(E_mm) = Var(STATE) + Var(FUNCTION) + Var(INTERACTION)
                 + 2 Cov(STATE, FUNCTION) + 2 Cov(STATE, INTERACTION)
                 + 2 Cov(FUNCTION, INTERACTION)
   (equivalently Var(E_mr) = Var(STATE) and Var(E_rm) = Var(FUNCTION), since E_rr is fixed).
   Report components and fractions (ddof=0, stated in attrs). Note in attrs:
   - the obs partition separates observed-state from ET-product uncertainty
   - Var(STATE) within trendy_s3 reflects LAI only (AI is shared)
   - trendy_s3 and ippe are offline (structural vs. parametric spread)
   - the lens partition is the internal-variability floor. Flag fppe/ippe components within the
     lens values.
5. Bin-count sensitivity (final labelled section of main()): read this script's outputs for every
   bins folder present under all/decomp/<period>/, and write a cross-bin summary to
   all/decomp/<period>/. Flag variance fractions that shift by more than 10 percentage points.
   Skip with a printed note if only one bin count exists.

## Implementation
- `etunc/decomposition.py`: small single-purpose functions, each with a concise comment stating
  its assumption:
  - p and f from a weighted bin_stats output
  - pooled dataset (with the source weights as an argument)
  - obs source weights (OBS_POOL_WEIGHTING)
  - counterfactual means E[x, y] on a common support
  - decompose(p_m, f_m, p_r, f_r), returning the four counterfactual means, the three terms and
    the off-support masses
  - range by product axis
  - variance_partition
- `scripts/decompose_state_function.py`, following the existing script conventions:
  - module-level settings (TRENDY_FAMILIES, OBS_PRODUCT_AXES, OBS_POOL_WEIGHTING)
  - cfg.summarize() and cfg.summarize_settings(globals())
  - a docstring listing every output
  - each comparison written out in main()
  - FileNotFoundError naming the missing bin-all-clim run (except trendy_s2, which is optional)
- pixi task `decompose-state-function` via scripts/run_logged.sh.
- Outputs, using a new output kind "decomp" under family "all":
  - cfg.output_dir(PROC_ROOT, "all", "decomp", period, bins):
    - decomp_vs_obs: variables E_rr, E_rm, E_mr, E_mm, FUNCTION, STATE, INTERACTION, E_m, E_r,
      OFF_SUPPORT_m, OFF_SUPPORT_r on dims (dataset, obs_reference).
      - `dataset` has a `family` coord.
      - `obs_reference` covers each obs source plus "obs_pooled", with the product coords of
        OBS_PRODUCT_AXES ("pooled" for obs_pooled).
      - Obs sources appear in `dataset` with family "obs" and their product coords.
    - decomp_obs_axis_range: (dataset, quantity, axis), the ranges of comparison 3, with axis
      over "all" plus each axis in OBS_PRODUCT_AXES that has more than one product.
    - decomp_variance: (family, component), against obs_pooled.
    - a tidy CSV (one row per dataset × obs_reference, including the product coords).
  - cfg.output_dir(PROC_ROOT, "all", "decomp", period): the cross-bin summary, dims
    (family, component, bins).
  - cfg.output_dir(FIG_ROOT, "all", "decomp", period, bins): per evaluated family, vs. obs_pooled:
    - primary panel: E_rr, E_rm, E_mr, E_mm per dataset
    - secondary panel: FUNCTION, STATE, INTERACTION per dataset, with E_mm − E_rr marked
    - error bars for the range across obs references
    - the obs-spread range of each term shaded
    - for fppe/ippe, the lens range shaded
    Plus one panel of the obs-axis ranges per quantity, which today has only the ET axis.
  - File names: all.et_clim.area_weighted.decomp_<vs_obs|obs_axis_range|variance>.qbin_pooled_all.<bins>.<grid>.<period>
- Docs:
  - add the script to "Analyses outside the pipeline" in CLAUDE.md and docs/binning-workflow.md
  - add "decomp" to the kinds listed under Output folders in CLAUDE.md
  - add the commands to the CLAUDE.md command block and the README tables
  - add tests/test_decomposition.py to the Tests list in CLAUDE.md
  - document that adding obs LAI/P/Rn products needs no change to this script: rerun the
    pipeline, then rerun it
  - while there, correct two stale statements:
    - CLAUDE.md and docs/binning-workflow.md say prep_trendy.py is not run by run_all.sh (it is,
      as S3)
    - the CLAUDE.md path for TRENDY_ROOT disagrees with etunc/config.py
    Report both corrections in the commit message.

## Tests (`tests/test_decomposition.py`, synthetic data built with binning.bin_stats(weights=...))
1. FUNCTION + STATE + INTERACTION == E_mm − E_rr to within 1e-10 relative error.
2. E_mm == E_m and E_rr == E_r when the support is complete.
3. Equal f → FUNCTION = INTERACTION = 0 and E_rm = E_rr.
   Equal p → STATE = INTERACTION = 0 and E_mr = E_rr.
4. p and f from weighted bin_stats reproduce sum a e / sum a. Using `count` for p does not, when
   a varies.
5. The pooled dataset gives E = mean_s E_s with unequal member counts per source.
6. Empty bins: off-support mass is correct, E_m − E_mm equals the ET of the off-support land, and
   the scalar quantities contain no NaNs.
7. The variance partition sums to Var(E_mm) against a fixed reference.
8. Shared AI: when members differ only in LAI (identical AI field), the AI marginal of p is
   identical across members.
9. Obs product axes, synthetic obs with two ET and two LAI products (full factorial):
   - sources that differ only in ET product give STATE = INTERACTION = 0 between them
   - sources that differ in LAI product give nonzero STATE
   - the axis ranges pick up each axis separately
   - an axis with one product is skipped
10. Non-factorial obs combinations: with OBS_POOL_WEIGHTING = "source", the reported effective
    product weights match the counts.

Run `pixi run test` before reporting completion.

## Interpretation (docstrings and attrs)
- Domain means are over the common area mask, not all land.
- The terms are reference-weighted by design: FUNCTION uses the observed state and STATE uses the
  observed response, so both inherit the reference's uncertainty (quantified in comparison 3).
- The observed state is sampled only as far as the obs sources vary in LAI, P and Rn. With a
  single product on an axis, that axis contributes no observational state uncertainty, and the
  obs spread understates it. The attrs record which axes vary.
- LAI and AI respond to ET in coupled runs (P recycling, Rn, prognostic LAI), so this is an
  accounting identity on a shared partition of a shared domain, not a causal attribution.
- In the offline families (ippe, TRENDY), P is prescribed. TRENDY's Rn is also prescribed, from
  ERA5, so its STATE term contains no climate feedback at all.
- FUNCTION includes every control not on the two axes (soil moisture, seasonality, VPD, rooting
  depth).