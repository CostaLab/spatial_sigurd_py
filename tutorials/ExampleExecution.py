# Importing packages and functions.
import gc
import math
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from matplotlib.colors import ListedColormap

import spatial_sigurd_py
from spatial_sigurd_py.pp.basic import _extract_xy_from_obsm

ad.settings.allow_write_nullable_strings = True


def plot_variants_per_coverage(coverage_df, sample_use, output_path):
    """
    Parameters
    ----------
    coverage_df : pd.DataFrame
        DataFrame with at least a "avg_coverage" column (one row per variant).
    sample_use : str
        Sample name, used for title and output filename.
    output_path : str or Path
        Output directory where the plot will be saved.
    """
    print("We get the number of variants with at least X coverage.", flush=True)

    # Build coverage threshold summary
    thresholds = list(range(5, 91, 5))  # [5, 10, ..., 90]
    variants = [(coverage_df["avg_coverage"] >= cov).sum() for cov in thresholds]

    variants_per_coverage = pd.DataFrame({"Coverage": thresholds, "Variants": variants})

    print("We plot the number of variants retained per coverage.", flush=True)

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=variants_per_coverage, x="Coverage", y="Variants", marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Variants")
    ax.set_title(f"Variants {sample_use}")

    # Save plot
    output_file = f"{output_path}/Variants_Per_Coverage_{sample_use}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


print("Parameters.", flush=True)
vaf_use = 0.1
alts_threshold = 2
sample_use = "my_sample"
coverage_use = 50
radius_factor_use = 1.15
design_matrix_path = "/path/to/DesignMatrix.csv"
reference_path = "/path/to/chrM_refAllele.txt"
positions_path = "/path/to/positions.csv"
meta_data_path = "/path/to/my/metadata.csv"
low_coverage_threshold = 10
minimum_fw_rev_reads = 2
minimum_strand_correlation = 0.65
minimum_cells_detected = 5

output = f"/path/to/my/output/MinVAF_{vaf_use:.2f}_MinAltReads_{alts_threshold}_Pruning/"
print("Using:", flush=True)
print("Sample:   " + sample_use, flush=True)
print("Coverage: " + str(coverage_use), flush=True)
print("Radius:   " + str(radius_factor_use), flush=True)


# We make the necessary directory.
if not os.path.isdir(f"{output}{sample_use}/Variant_On_Spatial/"):
    os.makedirs(f"{output}{sample_use}/Variant_On_Spatial/")
# end if statement


print("Setting scanpy output folder.", flush=True)
sc.settings.figdir = f"{output}{sample_use}"
sc.set_figure_params(dpi_save=150, figsize=[10, 10])


print("Loading the data.", flush=True)
adata = spatial_sigurd_py.readwrite.LoadingMGATK_typewise(
    design_matrix_path=design_matrix_path,
    patient=sample_use,
    reference_path=reference_path,
    samples_path=None,
    barcodes_path=None,
    patient_column="patient",
    type_use="ATAC",
    chromosome_prefix="chrM",
    min_cells=2,
    cells_include=None,
    cells_exclude=None,
    verbose=True,
)


print("Loading the meta data.", flush=True)
# First, we removed some spots during the annotation process.
# We load the positions and subset them accordingly.
positions = pd.read_csv(positions_path, header=None)
# Adding a prefix to be consistent.
positions[0] = sample_use + "_" + positions[0].astype(str) + "-1"
# keep only in-tissue spots
positions = positions[positions[1] == 1]
positions = positions[[0, 5, 4]]
# Renaming columns
positions.columns = ["CellID", "X", "Y"]
positions = positions[positions["CellID"].isin(adata.obs_names)]


meta_data = pd.read_csv(
    meta_data_path,
    index_col=0,
)
meta_data["Sample"] = [x[1:] for x in meta_data["Sample"]]
meta_data.index = [x[1:] for x in meta_data.index]

# Due to the reindexing, we would add spots that are not in the annotation.
# Then we distort the results. So, we first subset the positions to the
# annotated spots.
meta_data = meta_data[meta_data["Sample"] == sample_use]
positions = positions[positions["CellID"].isin(meta_data.index)]

# Align the rows (cells/spots)
meta_data = meta_data.reindex(adata.obs_names)
# Remove missing values if they exist.
meta_data = meta_data.loc[~meta_data.index.isna()]
adata.obs["CellType"] = meta_data["CellType"]


print("Filtering the data.", flush=True)
cells_include = set(positions["CellID"].dropna())
adata = spatial_sigurd_py.pp.Filtering(
    adata=adata,
    cells_include=cells_include,
    cells_exclude=None,
    fraction_threshold=vaf_use,
    keep_refs=True,
    alts_threshold=alts_threshold,
    min_cells_per_variant=None,
    min_variants_per_cell=None,
    verbose=True,
)

gc.collect()

# We add the positions to the annData object. Since we have already removed all
# spots that are not in positions and the meta data, we are not adding spots
# that should have been removed. This would happen with reindex since we would
# introduce NAs.
positions = positions.set_index("CellID").reindex(adata.obs_names)
adata.obsm["spatial"] = positions.set_index("CellID").loc[adata.obs_names, ["X", "Y"]].to_numpy(dtype=np.float64)


print("Performing the pruning.", flush=True)
adata_pruned = spatial_sigurd_py.pp.Pruning(
    adata=adata,
    spatial_key="spatial",
    x_col="X",
    y_col="Y",
    radius_factor=radius_factor_use,
    drop_zero_variants=True,
    drop_zero_cells=False,
    copy=True,
    verbose=True,
)


print("Plotting the annotation.", flush=True)
# Get the x/y coordinates from the annData object.
xy = _extract_xy_from_obsm(adata_pruned.obsm["spatial"])
x = xy[:, 0]
y = xy[:, 1]

color_map = {
    "CellType1": "#1f77b4",
    "CellType2": "#ff7f0e",
    "CellType3": "#2ca02c",
}
fallback_color = "#999999"

celltypes = sorted(adata_pruned.obs["CellType"].dropna().unique())

fig, ax = plt.subplots(figsize=(10, 10))
for ct in celltypes:
    sel = adata_pruned.obs["CellType"].to_numpy() == ct
    color = color_map.get(ct, fallback_color)
    ax.scatter(x[sel], y[sel], s=28, alpha=0.9, color=color, label=ct)

buffer = 50
ax.set_xlim(x.min() - buffer, x.max() + buffer)
ax.set_ylim(y.min() - buffer, y.max() + buffer)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend(
    title="Cell Type",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    markerscale=2,
    frameon=False,
    title_fontproperties={"weight": "bold", "size": 10},
    fontsize=8,
)
ax.set_title("CellType", fontsize=16, fontweight="bold")
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.85, top=0.97, bottom=0.01)
plt.savefig(f"{output}{sample_use}/SpatialPlot_CellType.png", dpi=150, bbox_inches="tight")
plt.close()


print("Plotting the coverage distribution.", flush=True)
avg_coverages = np.asarray(adata_pruned.var["avg_coverage"].values, dtype=float)
avg_coverages = avg_coverages[np.isfinite(avg_coverages)]
n = avg_coverages.size

q1, q3 = np.percentile(avg_coverages, [25, 75])
iqr = q3 - q1
bin_width = 2 * iqr / (n ** (1.0 / 3.0))

if bin_width <= 0:
    bins = 1
else:
    bins = max(1, math.ceil((avg_coverages.max() - avg_coverages.min()) / bin_width))

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(avg_coverages, bins=bins)
ax.set_xlabel("Average coverage")
ax.set_ylabel("Count")
ax.set_title("Histogram of average coverage")

out_path = f"{output}{sample_use}/{sample_use}_avg_coverage_hist.png"
fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close()


print("We get the number of variants with at least X coverage.", flush=True)
plot_variants_per_coverage(coverage_df=adata_pruned.var, sample_use=sample_use, output_path=f"{output}{sample_use}/")


print("Selecting variants.", flush=True)
vois = spatial_sigurd_py.tl.Variant_Selection_VMR(
    adata=adata_pruned,
    stabilize_variance=True,
    low_coverage_threshold=low_coverage_threshold,
    minimum_fw_rev_reads=minimum_fw_rev_reads,
    verbose=True,
)


print("Saving variants.", flush=True)
vois.to_csv(f"{output}{sample_use}/VOIs_All.csv")
vois = pd.read_csv(f"{output}{sample_use}/VOIs_All.csv", index_col=0)


print("We get the spatial clones.", flush=True)
vois_filtered = vois.query(
    "strand_correlation > @minimum_strand_correlation and mean_coverage > @coverage_use and cells_detected >= @minimum_cells_detected"
)
adata_pruned_subset = adata_pruned[:, vois_filtered.variant].copy()
adata_clones, variant_df, svg_list, variant_df_enhanced = spatial_sigurd_py.tl.Spatial_Clone_Detection(
    adata=adata_pruned_subset,
    spatial_key="spatial",
    filter_by_min_spots=5,
    normalize=True,
    neighbors_for_ratio=8,
    neighbors_for_ratio_absolute=True,
    detect_svg_normalize_lap=False,
    detect_svg_filter_peaks=True,
    detect_svg_S=1,
    detect_svg_cal_pval=True,
    p_value_threshold_svvs=0.05,
    perform_low_pass_enhancement=True,
    low_pass_enhancement_ratio_low_freq=15,
    low_pass_enhancement_c=0.003,
    low_pass_enhancement_normalize_lap=False,
    perform_ftu=True,
    ftu_resolution=(0.7, 1.8, 0.1),
    ftu_weight_by_freq=False,
    ftu_normalize_lap=False,
    neighbors_for_ftu=8,
    ftu_random_state=0,
    ftu_on_error="warn",
    verbose=True,
)


print("Plotting the SVGs.", flush=True)
sc.pl.spatial(adata_clones, color=svg_list, size=1.6, cmap="magma", use_raw=False, spot_size=25, save="_SVGs_Top8.png")


print("We plot the variants on the spatial plot using the original data.", flush=True)
coords = adata_clones.obsm["spatial"]
# Separate x and y for plotting
x = coords[:, 0]
y = coords[:, 1]
cmap = ListedColormap(["#B4B4B4", "#FF6879"])  # 0 as grey and 1 as light red
for variant_use in svg_list:
    print(variant_use, flush=True)
    count_values = adata_clones[:, variant_use].layers["X_original"]
    if sp.issparse(count_values):
        count_values = count_values.toarray().ravel()
    else:
        count_values = np.array(count_values).ravel()

    # Get raw values and binarize (>0 -> 1, else 0)
    binary = (count_values > 0).astype(int)
    order = np.argsort(count_values)

    # Create a figure, plot, and save
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[order], y[order], c=binary[order], cmap=cmap, s=5)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(variant_use, pad=4)
    ax.margins(x=0.01, y=0.01)

    filename = f"{output}{sample_use}/Variant_On_Spatial/{variant_use}_spatial.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


print("We plot the variants on the spatial plot using the enhanced data.", flush=True)
for variant_use in svg_list:
    print(variant_use, flush=True)
    count_values = adata_clones[:, variant_use].layers["X_enhanced"]
    if sp.issparse(count_values):
        count_values = count_values.toarray().ravel()
    else:
        count_values = np.array(count_values).ravel()

    # Get raw values and binarize (>0 -> 1, else 0)
    binary = (count_values > 0).astype(int)
    order = np.argsort(count_values)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[order], y[order], c=binary[order], cmap=cmap, s=5)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(variant_use, pad=4)
    ax.margins(x=0.01, y=0.01)

    filename = f"{output}{sample_use}/Variant_On_Spatial/{variant_use}_spatial_enhanced.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


print("Saving the output.", flush=True)
adata_clones.write_h5ad(f"{output}{sample_use}/adata_clones.h5ad", compression="gzip", compression_opts=9)
