import gc
import os
from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from cyvcf2 import VCF
from scipy.io import mmread


def _dataframe_to_coo_sparse(df: pd.DataFrame, dtype = np.float32) -> sp.coo_matrix:
    """Return a COO without densifying; cast to pandas-Sparse if needed."""
    fill = 0.0 if np.issubdtype(dtype, np.floating) else 0
    if not any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes):
        df = df.astype(pd.SparseDtype(np.dtype(dtype), fill), copy=False)
    else:
        # make sure subtype matches desired dtype
        df = df.astype(pd.SparseDtype(np.dtype(dtype), fill), copy=False)
    return df.sparse.to_coo()


# Padding and reordering matrices.
def _pad_and_cast(
    df: pd.DataFrame,
    rows,
    cols,
    desired_dtype: pd.SparseDtype,
):
    """
    Padding and casting the matrices to ensure that all variants are present.
      - df: your input object
      - rows: the cells
      - cols: the variants
      - desired_dtype: the desired type, for example pd.SparseDtype(np.int32, 0) or pd.SparseDtype(np.float32, 0.0)
    """
    # 1) Ensure sparse dtype with fill_value matching zeros
    if not isinstance(desired_dtype, pd.SparseDtype):
        raise TypeError("desired_dtype must be a pandas SparseDtype")
    
    df = df.astype(desired_dtype, copy=False)
    
    # 2) Add missing rows using a sparse zero block; avoid two-axis reindex
    if df.index.is_unique is False:
        df = df[~df.index.duplicated(keep="first")]
    
    if df.columns.is_unique is False:
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    
    target_rows = pd.Index(rows)
    target_cols = pd.Index(cols)
    
    missing_rows = target_rows.difference(df.index)
    if len(missing_rows):
        zero_block = sp.csr_matrix((len(missing_rows), df.shape[1]), dtype=desired_dtype.subtype)
        add_rows = pd.DataFrame.sparse.from_spmatrix(zero_block, index=missing_rows, columns=df.columns)
        df = pd.concat([df, add_rows], axis=0, copy=False)
    
    # 3) Add missing columns via one sparse zero block (not per-column)
    missing_cols = target_cols.difference(df.columns)
    if len(missing_cols):
        zero_block = sp.csr_matrix((df.shape[0], len(missing_cols)), dtype=desired_dtype.subtype)
        add_cols = pd.DataFrame.sparse.from_spmatrix(zero_block, index=df.index, columns=missing_cols)
        df = pd.concat([df, add_cols], axis=1, copy=False)
    
    # 4) Finally reorder both axes WITHOUT fill_value (everything exists now)
    # Doing order-only reindex avoids dense fill paths
    df = df.loc[target_rows, target_cols]
    
    # 5) Guarantee dtype (concat can sometimes widen)
    df = df.astype(desired_dtype, copy=False)
    return df


def _pad_sparse(df: pd.DataFrame, rows: pd.Index, cols: pd.Index, dtype) -> pd.DataFrame:
    # make a sparse-dtype frame first
    fill = 0.0 if np.issubdtype(dtype, np.floating) else 0
    sdf = df.astype(pd.SparseDtype(np.dtype(dtype), fill))
    # reindex remains sparse and pads with fill
    sdf = sdf.reindex(index=rows, columns=cols, fill_value=fill)
    # keep the dtype tight
    return sdf.astype(pd.SparseDtype(np.dtype(dtype), fill), copy=False)


def _subset(mat: sp.csr_matrix, keep: Sequence[bool]) -> sp.csr_matrix:
    return mat[:, keep]


def _pl_pos_to_pd(tbl: pl.DataFrame) -> pd.DataFrame:
    df_pd = tbl.to_pandas()
    df_pd.set_index("position", inplace=True)
    return df_pd.astype(np.int64, copy=False)


def _df_to_csc(df: pd.DataFrame, dtype=np.float32) -> sp.csc_matrix:
    """Convert DataFrame to sparse CSC without making an additional dense copy."""
    return sp.csc_matrix(df.to_numpy(dtype=dtype, copy=False)).T


def _df_to_csc_sparse_v2(df: pd.DataFrame, dtype=np.float32) -> sp.csc_matrix:
    """Convert Dense or pandas-Sparse DataFrame (variants×cells) to CSC (cells×variants) without densifying."""
    # We check if the dtype is sparse.
    if any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes):
        coo = df.sparse.to_coo()  # variants × cells ==> transposing back to cells x variants
        return coo.tocsc().astype(dtype, copy=False).T
    
    # Fallback if the dtype is dense.
    return sp.csc_matrix(df.to_numpy(dtype=dtype, copy=False)).T


def _df_to_csc_sparse(df: pd.DataFrame, dtype=np.float32) -> sp.csc_matrix:
    """
    Convert a (variants × cells) DataFrame to a CSC matrix (cells × variants)
    without densifying. Coerces to pandas-Sparse first to guarantee the sparse path.
    
    Parameters
    ----------
    df : pd.DataFrame
        variants × cells
    dtype : np.dtype
        target numeric dtype for the SciPy matrix (e.g., np.int32, np.float32)
    
    Returns
    -------
    scipy.sparse.csc_matrix
        cells × variants
    """
    # Fast path for empty shapes
    if df.shape[0] == 0 or df.shape[1] == 0:
        return sp.csc_matrix((df.shape[1], df.shape[0]), dtype=dtype)
    
    # Decide proper fill_value for SparseDtype
    fill = 0.0 if np.issubdtype(np.dtype(dtype), np.floating) else 0
    SD = pd.SparseDtype(np.dtype(dtype), fill)
    
    # Coerce to pandas-Sparse no matter what the current dtype is.
    # This avoids the dense .to_numpy() code path entirely.
    if not any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes):
        df = df.astype(SD, copy=False)
    else:
        # Ensure subtype/fill match our target
        df = df.astype(SD, copy=False)
    
    # Sparse conversion: variants × cells (COO) -> CSC, then transpose to cells × variants
    coo = df.sparse.to_coo()
    return coo.tocsc().astype(dtype, copy=False).T


# Helper converting Polars to Pandas int32
def _pl_to_pd_int32(tbl: pl.DataFrame) -> pd.DataFrame:
    df_pd = tbl.to_pandas()
    df_pd.set_index("mutation", inplace=True)
    return df_pd.astype(np.int32, copy=False)


# Helper converting Polars to Pandas int64
def _pl_to_pd_int64(tbl: pl.DataFrame) -> pd.DataFrame:
    df_pd = tbl.to_pandas()
    df_pd.set_index("mutation", inplace=True)
    return df_pd.astype(np.int64, copy=False)


# Helper converting Polars to Pandas float32
def _pl_to_pd_float32(tbl: pl.DataFrame) -> pd.DataFrame:
    df_pd = tbl.to_pandas()
    df_pd.set_index("mutation", inplace=True)
    return df_pd.astype(np.float32, copy=False)


def _pl_to_pd_float64(tbl: pl.DataFrame) -> pd.DataFrame:
    df_pd = tbl.to_pandas()
    df_pd.set_index("mutation", inplace=True)
    return df_pd.astype(np.float64, copy=False)


def _align_cols(df: pd.DataFrame, cols: list[str], dtype=None) -> pd.DataFrame:
    out = df.reindex(columns=cols, fill_value=0)
    if dtype is not None:
        out = out.astype(dtype, copy=False)

    return out


def compute_weighted_qualities(
    alt_fwd: pd.DataFrame,
    alt_fwd_qual: pd.DataFrame,
    alt_rev: pd.DataFrame,
    alt_rev_qual: pd.DataFrame,
    ref_fwd: pd.DataFrame,
    ref_fwd_qual: pd.DataFrame,
    ref_rev: pd.DataFrame,
    ref_rev_qual: pd.DataFrame,
    *,
    alt_reads: pd.DataFrame | None = None,
    ref_reads: pd.DataFrame | None = None,
    check_alignment: bool = False,
) -> dict:
    """
    Compute read-weighted average base qualities.
    
    Returns a dict with:
      - alt_qual_global,      ref_qual_global,      all_qual_global     : DataFrame (variants × cells, float32)
      - alt_qual_per_variant, ref_qual_per_variant, all_qual_per_variant: Series (float32)
      - alt_qual_per_cell,    ref_qual_per_cell,    all_qual_per_cell   : Series (float32)
    
    Memory-lean strategy:
      - Keep inputs and per-cell/per-variant results in float32.
      - Do large reductions (sum across many cells/variants) in float64 for stability.
    """
    # We check if the matrices are actually aligned (equal). If not, we throw an error.
    # Since we already should have done this before, this is simply an additional
    # sanity check.
    if check_alignment:
        idx = alt_fwd.index
        cols = alt_fwd.columns
        for df in (alt_fwd_qual, alt_rev, alt_rev_qual, ref_fwd, ref_fwd_qual, ref_rev, ref_rev_qual):
            if not (df.index.equals(idx) and df.columns.equals(cols)):
                raise RuntimeError("Input matrices are not aligned in variants/cells.")
    
    # Cast counts to float32 for multiplications
    alt_fwd_float32 = alt_fwd.astype(np.float32, copy=False)
    alt_rev_float32 = alt_rev.astype(np.float32, copy=False)
    ref_fwd_float32 = ref_fwd.astype(np.float32, copy=False)
    ref_rev_float32 = ref_rev.astype(np.float32, copy=False)
    
    # The qualities should already be float32.
    # Here, we ensure this.
    alt_fwd_qual_float32 = alt_fwd_qual.astype(np.float32, copy=False)
    alt_rev_qual_float32 = alt_rev_qual.astype(np.float32, copy=False)
    ref_fwd_qual_float32 = ref_fwd_qual.astype(np.float32, copy=False)
    ref_rev_qual_float32 = ref_rev_qual.astype(np.float32, copy=False)
    
    # Weighted numerators (sum of count * per-strand-avg-quality)
    alt_qsum = alt_fwd_float32 * alt_fwd_qual_float32 + alt_rev_float32 * alt_rev_qual_float32  # float32
    ref_qsum = ref_fwd_float32 * ref_fwd_qual_float32 + ref_rev_float32 * ref_rev_qual_float32  # float32
    
    # Denominators (counts) alternative
    if alt_reads is None:
        alt_reads = (alt_fwd + alt_rev).astype(np.int32, copy=False)
    # end if statement
    
    # Denominators (counts) reference
    if ref_reads is None:
        ref_reads = (ref_fwd + ref_rev).astype(np.int32, copy=False)
    # end if statement
    
    # Convert denominators to float32 and mask zeros as NaN to avoid divide-by-zero
    alt_denominator = alt_reads.astype(np.float32, copy=False).where(alt_reads != 0)
    ref_denominator = ref_reads.astype(np.float32, copy=False).where(ref_reads != 0)
    all_denominator_counts = alt_reads + ref_reads
    all_denominator = all_denominator_counts.astype(np.float32, copy=False).where(all_denominator_counts != 0)
    
    # Per-variant × per-cell weighted qualities
    alt_qual_global = (alt_qsum / alt_denominator).astype(np.float32)
    ref_qual_global = (ref_qsum / ref_denominator).astype(np.float32)
    all_qual_global = ((alt_qsum + ref_qsum) / all_denominator).astype(np.float32)
    
    # -------- Weighted summaries --------
    # Row-wise (per-variant): sum over cells with float64 accumulator
    # Here, we use float64 to ensure that the calculations are accurate.
    alt_qsum_row = np.sum(alt_qsum.values, axis=1, dtype=np.float64)
    alt_den_row = np.sum(alt_reads.values, axis=1, dtype=np.float64)
    ref_qsum_row = np.sum(ref_qsum.values, axis=1, dtype=np.float64)
    ref_den_row = np.sum(ref_reads.values, axis=1, dtype=np.float64)
    
    all_qsum_row = alt_qsum_row + ref_qsum_row
    all_den_row = alt_den_row + ref_den_row
    
    # Avoid divide-by-zero with np.where
    alt_qual_per_variant = pd.Series(
        alt_qsum_row / np.where(alt_den_row == 0, np.nan, alt_den_row), index=alt_reads.index, dtype=np.float32
    )
    ref_qual_per_variant = pd.Series(
        ref_qsum_row / np.where(ref_den_row == 0, np.nan, ref_den_row), index=ref_reads.index, dtype=np.float32
    )
    all_qual_per_variant = pd.Series(
        all_qsum_row / np.where(all_den_row == 0, np.nan, all_den_row), index=alt_reads.index, dtype=np.float32
    )
    
    # Column-wise (per-cell): sum over variants with float64 accumulator
    alt_qsum_col = np.sum(alt_qsum.values, axis=0, dtype=np.float64)
    alt_den_col = np.sum(alt_reads.values, axis=0, dtype=np.float64)
    ref_qsum_col = np.sum(ref_qsum.values, axis=0, dtype=np.float64)
    ref_den_col = np.sum(ref_reads.values, axis=0, dtype=np.float64)
    
    all_qsum_col = alt_qsum_col + ref_qsum_col
    all_den_col = alt_den_col + ref_den_col
    
    alt_qual_per_cell = pd.Series(
        alt_qsum_col / np.where(alt_den_col == 0, np.nan, alt_den_col), index=alt_reads.columns, dtype=np.float32
    )
    ref_qual_per_cell = pd.Series(
        ref_qsum_col / np.where(ref_den_col == 0, np.nan, ref_den_col), index=ref_reads.columns, dtype=np.float32
    )
    all_qual_per_cell = pd.Series(
        all_qsum_col / np.where(all_den_col == 0, np.nan, all_den_col), index=alt_reads.columns, dtype=np.float32
    )
    
    return {
        "alt_qual_global": alt_qual_global,
        "ref_qual_global": ref_qual_global,
        "all_qual_global": all_qual_global,
        "alt_qual_per_variant": alt_qual_per_variant,
        "ref_qual_per_variant": ref_qual_per_variant,
        "all_qual_per_variant": all_qual_per_variant,
        "alt_qual_per_cell": alt_qual_per_cell,
        "ref_qual_per_cell": ref_qual_per_cell,
        "all_qual_per_cell": all_qual_per_cell,
    }


def load_vartrix_typewise(
    samples_file: str | Path,
    *,
    samples_path: str | Path | None = None,
    barcodes_path: str | Path | None = None,
    snp_path: str | Path | None = None,
    vcf_path: str | Path,
    patient: str,
    patient_column: str = "patient",
    type_use: str = "scRNAseq_Somatic",
    min_reads: int | None = None,
    min_cells: int = 2,
    cells_include: Sequence[str] | None = None,
    cells_exclude: Sequence[str] | None = None,
    cellbarcode_length: int = 18,
    verbose: bool = True,
) -> ad.AnnData | None:
    """
    Load VarTrix results type‑wise, filter and assemble into an AnnData object.
    
    Returns None if after filtering there are zero cells or zero variants.
    """
    # We check if we want to load a single file or multiple.
    # We check if the variables are not None. In that case they are considered
    # "True".
    if samples_path and barcodes_path and snp_path:
        if verbose:
            print(f"Loading the data for sample {patient}.")
        # end if statement
        
        df = pd.DataFrame(
            {
                "patient": [patient],
                "sample": [patient],
                "input_path": [str(samples_path)],
                "cells": [str(barcodes_path)],
            }
        )
        samples = [patient]
    else:
        if verbose:
            print(f"Loading the data for patient {patient}.")
            print("Reading the design matrix.")
        # end if statement
        
        df = pd.read_csv(samples_file, dtype=str)
        
        if patient_column not in df.columns and patient_column != "merge":
            raise ValueError(f"Column '{patient_column}' not found in design matrix.")
        # end if statement
        
        # Keep only rows with the source vartrix.
        # We ignore capitalization. NA values are returned as False.
        df = df[df["source"].str.contains("vartrix", case=False, na=False)]
        # If we want to merge the samples, we check this here.
        if patient_column != "merge":
            df = df[df[patient_column] == patient]
        # end if statement
        
        df = df[df["type"] == type_use]
        samples = df["sample"].tolist()
    # end if statement
    
    # We get the list of SNPs and the barcodes.
    #if verbose:
    #    print("Loading SNV loci files.")
    #if snp_path:
    #    paths_snps = [str(snp_path)] * len(samples)
    #else:
    #    paths_snps = [str(Path(row.input_path) / "SNV.loci.txt") for _, row in df.iterrows()]
    # end if statement
    
    # snps_list = {s: pd.read_table(p, header=None, names=["locus"]) for s, p in zip(samples, paths_snps, strict=True)}
    
    if verbose:
        print("Loading cell‑barcode files.")
    barcodes = {
        s: pd.read_table(str(row.cells), header=None, names=["barcode"]) for s, row in df.set_index("sample").iterrows()
    }
    
    # 3. read VCF and build variant names
    if verbose:
        print("Loading VCF.")
    vcf = VCF(str(vcf_path))
    infos = []
    for rec in vcf:
        info = rec.INFO
        if {"GENE", "AA", "CDS"}.issubset(info):
            infos.append(f"{info['GENE']}_{info['AA']}_{info['CDS']}")
        else:
            # fall back to record ID, sanitized
            uid = rec.ID or f"{rec.CHROM}_{rec.POS}"
            infos.append(uid.replace(":", "_").replace("/", "_").replace("?", "_"))
        # end if statement
    # end for loop
    
    new_names = np.array(infos, dtype=str)
    
    # 4. load and rename sparse matrices per sample
    if verbose:
        print("Reading sparse genotype matrices.")
    cov_mats, ref_mats, con_mats = {}, {}, {}
    for idx, row in enumerate(df.itertuples(), 1):
        sample = row.sample
        inp = Path(row.input_path)
        if verbose:
            print(f"  Sample {idx}/{len(df)}: {sample}")
        # end if statement
        
        cm = mmread(inp / sample / "out_matrix_coverage.mtx").tocsr()
        rm = mmread(inp / sample / "ref_matrix_coverage.mtx").tocsr()
        xm = mmread(inp / sample / "out_matrix_consensus.mtx").tocsr()
        
        # prepend sample to barcodes
        bc = barcodes[sample]["barcode"].astype(str).tolist()
        cols = [f"{sample}_{b}" for b in bc]
        cm._shape = (cm.shape[0], len(cols))
        rm._shape = (rm.shape[0], len(cols))
        xm._shape = (xm.shape[0], len(cols))
        
        _ = cm.indices  # force indices to initialize. We assign the values to _ since ruff will complain otherwise.
        cm = sp.csr_matrix(cm)
        rm = sp.csr_matrix(rm)
        xm = sp.csr_matrix(xm)
        # assign names via AnnData later
        
        cov_mats[sample] = cm
        ref_mats[sample] = rm
        con_mats[sample] = xm
    # end for loop
    
    # 5. concat across samples
    if verbose:
        print("Merging matrices across samples…")
    coverage = sp.hstack(list(cov_mats.values()), format="csr")
    reference = sp.hstack(list(ref_mats.values()), format="csr")
    consensus = sp.hstack(list(con_mats.values()), format="csr")
    del cov_mats, ref_mats, con_mats
    gc.collect()
    
    # 6. cell filtering
    cols = sum((cov.columns.tolist() for cov in barcodes.values()), [])
    col_labels = [f"{s}_{b}" for s in samples for b in barcodes[s]["barcode"]]
    
    if cells_include is not None:
        if verbose:
            print("Applying include‑list filter…")
        mask = [c in set(cells_include) for c in col_labels]
        
        if not any(mask):
            raise ValueError("No cells left after include filter.")
        # end if statement
        
        coverage = _subset(coverage, mask)
        reference = _subset(reference, mask)
        consensus = _subset(consensus, mask)
    # end if statement
    
    if cells_exclude is not None:
        if verbose:
            print("Applying exclude‑list filter…")
        mask = [c not in set(cells_exclude) for c in col_labels]
        
        if not any(mask):
            raise ValueError("No cells left after exclude filter.")
        # end if statement
        
        coverage = _subset(coverage, mask)
        reference = _subset(reference, mask)
        consensus = _subset(consensus, mask)
    # end if statement
    
    # 7. apply min_reads threshold
    if min_reads is not None:
        if verbose:
            print(f"Zeroing reads < {min_reads} and recomputing consensus…")
        # end if statement
        
        # zero out low reads in‑place
        reference.data[reference.data < min_reads] = 0
        coverage.data[coverage.data < min_reads] = 0
        
        # binary indicators
        ref_bin = reference.copy()
        ref_bin.data = np.ones_like(ref_bin.data)
        
        cov_bin = coverage.copy()
        cov_bin.data = np.where(cov_bin.data > 0, 2, 0)
        
        consensus = ref_bin + cov_bin
        del ref_bin, cov_bin
        gc.collect()
    # end if statement
    
    # 8. re‑label rows & check dims
    if coverage.shape[0] != len(new_names):
        raise ValueError(f"Matrix rows ({coverage.shape[0]}) ≠ number of variants ({len(new_names)}).")
    # end if statement
    
    # build obs and var
    obs = pd.DataFrame(
        {
            "Cell": col_labels,
            "Patient": patient,
            "Sample": [c.split("_", 1)[0] for c in col_labels],
            "Type": type_use,
            **{"AverageCoverage": np.asarray(coverage.mean(axis=0)).ravel()},
        },
        index=col_labels,
    )
    
    var = pd.DataFrame(
        {"VariantName": new_names, **{"Depth": np.asarray((coverage + reference).mean(axis=1)).ravel()}},
        index=new_names,
    )
    
    # 9. drop low‑support variants & empty cells
    if verbose:
        print(f"Filtering variants in < {min_cells} cells…")
    
    keep_var = np.array((consensus >= 1).sum(axis=1)).ravel() >= min_cells
    consensus = consensus[keep_var, :]
    coverage = coverage[keep_var, :]
    reference = reference[keep_var, :]
    var = var.loc[new_names[keep_var], :]
    
    if verbose:
        print("Filtering cells with all zeros.")
    
    keep_cell = np.array((consensus > 0).sum(axis=0)).ravel() > 0
    consensus = consensus[:, keep_cell]
    coverage = coverage[:, keep_cell]
    reference = reference[:, keep_cell]
    obs = obs.iloc[keep_cell, :]
    
    if consensus.shape[0] == 0 or consensus.shape[1] == 0:
        if verbose:
            print(f"No data left (variants = {consensus.shape[0]}, cells = {consensus.shape[1]}). Returning None.")
        # end if statement
        return None
    # end if statement
    
    # 10. build fraction layer
    if verbose:
        print("Computing fraction layer.")
    # end if statement
    
    reads = coverage + reference
    # elementwise fraction on nonzero coverage positions
    cov_coo = coverage.tocoo()
    reads_vals = reads[cov_coo.row, cov_coo.col].A1
    frac_data = cov_coo.data / reads_vals
    fraction = sp.coo_matrix((frac_data, (cov_coo.row, cov_coo.col)), shape=cov_coo.shape).tocsr()
    
    # 11. assemble AnnData
    if verbose:
        print("Assembling AnnData…")
    # end if statement
    
    adata = ad.AnnData(X=consensus, obs=obs, var=var)
    adata.layers["fraction"] = fraction
    adata.layers["coverage"] = reads
    adata.layers["alts"] = coverage
    adata.layers["refs"] = reference
    
    return adata


def LoadingMGATK_typewise(
    patient: str,
    samples_path: Path | None = None,
    barcodes_path: Path | None = None,
    samples_file: str | None = None,
    patient_column: str = "patient",
    reference_path: str | Path | None = None,
    chromosome_prefix: str = "chrM",
    type_use: str = "ATAC",
    cells_include: set[str] | None = None,
    cells_exclude: set[str] | None = None,
    min_cells: int = 5,
    verbose: bool = True,
):
    """Memory‑efficient re‑implementation of the original mgatk loading code.
    
    Key optimisations
    • **Low‑width dtypes** – counts and positions use ``int32`` / ``uint32`` instead of the Pandas default ``int64``.
    • **Stay in Polars** for heavy joins/pivots; convert to Pandas only once per table.
    • **Sparse matrices** – all large 2‑D layers (``X`` and ``layers``) are stored as ``scipy.sparse.csc_matrix``.
    • **Float32 everywhere** for fractional layers and other floating‑point data.
    • *Aggressive* `del` + ``gc.collect`` helps release intermediate buffers promptly.
    """
    # 1. Here, we load the design matrix.
    if samples_path and barcodes_path:
        if verbose:
            print(f"Loading single sample {patient}.", flush=True)
        
        df = pd.DataFrame(
            {
                "patient": [patient],
                "sample": [patient],
                "input_path": [str(samples_path)],
                "cells": [str(barcodes_path)],
            }
        )
    else:
        if verbose:
            print(f"Loading samples for patient {patient}.", flush=True)
        
        df = pd.read_csv(samples_file, dtype=str)
        if (patient_column not in df.columns) and (patient_column != "merge"):
            raise ValueError(f"Column '{patient_column}' not found in design matrix.")
        
        df = df[df["source"].str.contains("mgatk", case=False, na=False)]
        if patient_column != "merge":
            df = df[df[patient_column] == patient]
        
        df = df[df["type"] == type_use]
    
    # 2. Here, we get the reference genome positions once.
    if verbose:
        print("Reading the reference.", flush=True)
    
    ref_df = pd.read_csv(
        reference_path,
        sep="\t",
        header=None,
        names=["position", "ref_base"],
        dtype={"position": np.int64},
    )
    ref_df["ref_base"] = ref_df["ref_base"].str.upper()
    
    muts = [
        f"{chromosome_prefix}_{pos}_{ref}>{alt}"
        for pos, ref in zip(ref_df["position"], ref_df["ref_base"], strict=True)
        for alt in ("A", "C", "G", "T")
        if alt != ref
    ]
    mut_positions = [int(m.split("_")[1]) for m in muts]
    
    alt_fwd_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int64, 0))
    alt_rev_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int64, 0))
    ref_fwd_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int64, 0))
    ref_rev_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int64, 0))
    coverage_pos_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int64, 0))
    
    # Polars copy for fast joins later
    ref_pl = pl.DataFrame(ref_df)
    
    # 3. Iterate over the samples
    # Schema to load the base file in polars.
    schema_common = {
        "position": pl.Int64,
        "cell": pl.Utf8,
        "fwd": pl.UInt64,
        "rev": pl.UInt64,
    }
    # Schema to load the coverage file in polars.
    schema_cov = {"position": pl.Int64, "cell": pl.Utf8, "cov": pl.UInt64}
    for _, row in df.iterrows():
        sample_use = row["sample"]
        folder = Path(row["input_path"])
        
        if verbose:
            print(f"Processing sample: {sample_use}.", flush=True)
        
        # 3a barcodes
        if verbose:
            print("Loading cell barcodes.", flush=True)
        
        cell_barcodes = (
            pl.read_csv(
                str(row["cells"]),
                has_header=False,
                separator="\t",
                new_columns=["cell"],
            )
            .select("cell")
            .to_series()
            .to_list()
        )
        
        if cells_include is not None:
            cell_barcodes = [c for c in cell_barcodes if c in cells_include]
        
        if cells_exclude is not None:
            cell_barcodes = [c for c in cell_barcodes if c not in cells_exclude]
        
        if not cell_barcodes:
            if verbose:
                print(f"No barcodes left after cell barcode filtering. Skipping sample {sample_use}.", flush=True)
            
            continue
        
        # 3b – read base files
        if verbose:
            print("Reading A/C/G/T base files.", flush=True)
        
        dfs: list[pl.DataFrame] = []
        for base in ("A", "C", "G", "T"):
            path_use = folder / f"{sample_use}.{base}.txt.gz"
            if verbose:
                print(f"{path_use.name}", flush=True)
            dfs.append(
                pl.read_csv(
                    str(path_use),
                    has_header=False,
                    separator=",",
                    new_columns=["position", "cell", "fwd", "rev"],
                    schema=schema_common,
                )
                .with_columns(pl.lit(base).alias("alt"))
                .filter(pl.col("cell").is_in(cell_barcodes))
            )
        
        all_bases = pl.concat(dfs)
        del dfs
        gc.collect()
        
        # 3c – ALT tables
        if verbose:
            print("Building ALT tables.", flush=True)
        
        alt_pl = (
            all_bases.join(ref_pl, on="position")
            .filter(pl.col("alt") != pl.col("ref_base"))
            .with_columns(
                pl.concat_str(
                    [
                        pl.lit(chromosome_prefix),
                        pl.lit("_"),
                        pl.col("position").cast(pl.Utf8),
                        pl.lit("_"),
                        pl.col("ref_base"),
                        pl.lit(">"),
                        pl.col("alt"),
                    ]
                ).alias("mutation")
            )
        )
        grp_alt = alt_pl.group_by(["mutation", "cell"]).agg(
            [pl.col("fwd").sum().alias("fwd"), pl.col("rev").sum().alias("rev")]
        )
        del alt_pl
        gc.collect()
        
        alt_fwd = grp_alt.pivot(values="fwd", index="mutation", on="cell", aggregate_function="first").fill_null(0)
        alt_rev = grp_alt.pivot(values="rev", index="mutation", on="cell", aggregate_function="first").fill_null(0)
        del grp_alt
        gc.collect()
        
        alt_fwd_pd = _pl_to_pd_int64(alt_fwd)
        alt_fwd_pd.columns = [f"{sample_use}_{c}" for c in alt_fwd_pd.columns]
        alt_rev_pd = _pl_to_pd_int64(alt_rev)
        alt_rev_pd.columns = [f"{sample_use}_{c}" for c in alt_rev_pd.columns]
        
        alt_fwd_pd = alt_fwd_pd.reindex(index=muts, fill_value=0)
        alt_rev_pd = alt_rev_pd.reindex(index=muts, fill_value=0)
        
        alt_fwd_global = alt_fwd_global.join(alt_fwd_pd, how="outer").fillna(0).astype(np.int64, copy=False)
        alt_rev_global = alt_rev_global.join(alt_rev_pd, how="outer").fillna(0).astype(np.int64, copy=False)
        del alt_fwd_pd, alt_rev_pd
        gc.collect()
        
        # 3d reference reads
        if verbose:
            print("Adding reference reads.", flush=True)
        
        ref_reads_pl = all_bases.join(ref_pl, on="position").filter(pl.col("alt") == pl.col("ref_base"))
        grp_ref = ref_reads_pl.group_by(["position", "cell"]).agg(
            [pl.col("fwd").sum().alias("fwd"), pl.col("rev").sum().alias("rev")]
        )
        del ref_reads_pl
        gc.collect()
        ref_fwd_pos = grp_ref.pivot(values="fwd", index="position", on="cell", aggregate_function="first").fill_null(0)
        ref_rev_pos = grp_ref.pivot(values="rev", index="position", on="cell", aggregate_function="first").fill_null(0)
        del grp_ref
        gc.collect()
        
        ref_fwd_pos_pd = _pl_pos_to_pd(ref_fwd_pos)
        ref_fwd_pos_pd.columns = [f"{sample_use}_{c}" for c in ref_fwd_pos_pd.columns]
        ref_rev_pos_pd = _pl_pos_to_pd(ref_rev_pos)
        ref_rev_pos_pd.columns = [f"{sample_use}_{c}" for c in ref_rev_pos_pd.columns]
        
        ref_fwd_pd = (
            ref_fwd_pos_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(np.int64, copy=False)
        )
        ref_rev_pd = (
            ref_rev_pos_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(np.int64, copy=False)
        )
        ref_fwd_global = ref_fwd_global.join(ref_fwd_pd, how="outer").fillna(0).astype(np.int64, copy=False)
        ref_rev_global = ref_rev_global.join(ref_rev_pd, how="outer").fillna(0).astype(np.int64, copy=False)
        del ref_fwd_pd, ref_rev_pd, ref_fwd_pos_pd, ref_rev_pos_pd
        gc.collect()
        
        if verbose:
            print("Calculating the coverage.")
        
        # Here, we use the coverage from the coverage file.
        path_use = folder / f"{sample_use}.coverage.txt.gz"
        cov_pl = pl.read_csv(
            str(path_use),
            has_header=False,
            separator=",",
            new_columns=["position", "cell", "cov"],
            schema=schema_cov,
        ).filter(pl.col("cell").is_in(cell_barcodes))
        cov_samp = cov_pl.pivot(values="cov", index="position", on="cell", aggregate_function="first").fill_null(0)
        del cov_pl
        gc.collect()
        cov_pd = _pl_pos_to_pd(cov_samp)
        cov_pd.columns = [f"{sample_use}_{c}" for c in cov_pd.columns]
        cov_pd = (
            cov_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(np.int64, copy=False)
        )
        coverage_pos_global = coverage_pos_global.join(cov_pd, how="outer").fillna(0).astype(np.int64, copy=False)
        del cov_pd, cov_samp
        gc.collect()
    
    # 4. Post‑processing
    # The columns were not in the same order in the different data.frames.
    # We correct this here.
    
    if verbose:
        print("Aligning columns and rows.", flush=True)
    
    # We put the matrixes in a list. Then we can conveniently iterate over them.
    dfs = {
        "alt_fwd_global": alt_fwd_global,
        "alt_rev_global": alt_rev_global,
        "ref_fwd_global": ref_fwd_global,
        "ref_rev_global": ref_rev_global,
        "coverage_pos_global": coverage_pos_global,
    }
    # Using the alt_fwd_global dataframe as a reference.
    reference_name, reference_df = next(iter(dfs.items()))
    reference_rows = reference_df.index
    reference_cols = reference_df.columns
    
    # Build cell/variant unions (preserve reference order; append extras in encounter order)
    cells_union = reference_rows
    for df_use in dfs.values():
        cells_union = cells_union.union(df_use.index, sort=False)
    
    variant_union = reference_cols
    for df_use in dfs.values():
        variant_union = variant_union.union(df_use.columns, sort=False)
    
    # Decide expected dtype per frame (qualities are float32; counts/coverage are int32)
    _expected_dtype = {
        name: (pd.SparseDtype(np.float64, 0.0) if "qual" in name else pd.SparseDtype(np.int64, 0))
        for name in dfs.keys()
    }
    
    # Pad every matrix
    for name in list(dfs.keys()):
        dfs[name] = _pad_and_cast(dfs[name], cells_union, variant_union, _expected_dtype[name])
    
    # Integrity check (all shapes/labels identical now)
    _rows = cells_union
    _cols = variant_union
    for name, df_use in dfs.items():
        if not (df_use.index.equals(_rows) and df_use.columns.equals(_cols)):
            raise RuntimeError(f"Alignment failed for {name}.")
    
    # Now, we unpack the list again to use the reordered data.
    alt_fwd_global = dfs["alt_fwd_global"]
    alt_rev_global = dfs["alt_rev_global"]
    ref_fwd_global = dfs["ref_fwd_global"]
    ref_rev_global = dfs["ref_rev_global"]
    coverage_pos_global = dfs["coverage_pos_global"]
    del (
        dfs,
        reference_name,
        reference_df,
        reference_rows,
        reference_cols,
        df_use,
    )
    gc.collect()
    
    if verbose:
        print(f"Filtering variants with less than {min_cells} cells.", flush=True)
    
    alt_reads = alt_fwd_global.astype(pd.SparseDtype(np.int64, 0)) + alt_rev_global.astype(pd.SparseDtype(np.int64, 0))
    nonzero_cells = alt_reads.astype(bool).sum(axis=1).astype(np.int64)
    keep_mut = nonzero_cells >= min_cells
    
    alt_reads = alt_reads.loc[keep_mut]
    alt_fwd_global = alt_fwd_global.loc[keep_mut]
    alt_rev_global = alt_rev_global.loc[keep_mut]
    ref_fwd_global = ref_fwd_global.loc[keep_mut]
    ref_rev_global = ref_rev_global.loc[keep_mut]
    coverage_pos_global = coverage_pos_global.loc[keep_mut]
    ref_reads = ref_fwd_global.astype(pd.SparseDtype(np.int64, 0)) + ref_rev_global.astype(pd.SparseDtype(np.int64, 0))
    
    coverage_pos_global = coverage_pos_global.astype(pd.SparseDtype(np.int64, 0), copy=False)
    row_sums = coverage_pos_global.to_numpy().sum(axis=1).astype(np.float64)
    
    mean_cov_per_variant = pd.Series(
        row_sums / coverage_pos_global.shape[1], index=coverage_pos_global.index, dtype=np.float64
    )
    
    alt_arr = alt_reads.to_numpy(dtype=np.float64, copy=False)
    cov_arr = coverage_pos_global.to_numpy(dtype=np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction_arr = np.divide(alt_arr, cov_arr, out=np.zeros_like(alt_arr, dtype=np.float64), where=cov_arr != 0)
    
    fraction = pd.DataFrame(
        fraction_arr,
        index=alt_reads.index,
        columns=alt_reads.columns,
        dtype=np.float64,
    )
    
    strand_concordance_reads_fwd = alt_fwd_global.to_numpy(dtype=np.float64, copy=False)
    strand_concordance_reads_rev = alt_rev_global.to_numpy(dtype=np.float64, copy=False)
    strand_concordance_reads_fwd_mean = strand_concordance_reads_fwd.mean(axis=1, keepdims=True)
    strand_concordance_reads_rev_mean = strand_concordance_reads_rev.mean(axis=1, keepdims=True)
    
    pearson_numerator = np.sum(
        (strand_concordance_reads_fwd - strand_concordance_reads_fwd_mean)
        * (strand_concordance_reads_rev - strand_concordance_reads_rev_mean),
        axis=1,
    )
    pearson_denomiator = np.sqrt(
        np.sum((strand_concordance_reads_fwd - strand_concordance_reads_fwd_mean) ** 2, axis=1)
        * np.sum((strand_concordance_reads_rev - strand_concordance_reads_rev_mean) ** 2, axis=1)
    )
    strand_concordance_pearson_correlation = np.divide(
        pearson_numerator, pearson_denomiator, out=np.zeros_like(pearson_numerator), where=pearson_denomiator != 0
    )
    
    strand_concordance = pd.Series(strand_concordance_pearson_correlation, index=alt_fwd_global.index, dtype=np.float64)
    
    del (
        strand_concordance_reads_fwd,
        strand_concordance_reads_rev,
        strand_concordance_reads_fwd_mean,
        strand_concordance_reads_rev_mean,
        pearson_numerator,
        pearson_denomiator,
        strand_concordance_pearson_correlation,
    )
    gc.collect()
    
    if alt_fwd_global.empty or alt_fwd_global.shape[1] == 0:
        if verbose:
            print(
                f"Filtering left {alt_fwd_global.shape[0]} variants × {alt_fwd_global.shape[1]} cells; returning None.",
                flush=True,
            )
        
        return None
    
    if verbose:
        print("Creating AnnData object.", flush=True)
    
    obs = pd.DataFrame(index=alt_reads.columns)
    var = pd.DataFrame(index=alt_reads.index)
    var["strand_concordance"] = strand_concordance.loc[var.index]
    
    adata = ad.AnnData(X=_df_to_csc_sparse(alt_reads, dtype=np.int64), obs=obs, var=var)
    adata.var["avg_coverage"] = mean_cov_per_variant.loc[adata.var_names]
    
    # All layers sparse (float64)
    adata.layers["fraction"] = _df_to_csc_sparse(fraction, dtype=np.float64)
    adata.layers["alt_reads"] = _df_to_csc_sparse(alt_reads, dtype=np.int64)
    adata.layers["ref_reads"] = _df_to_csc_sparse(ref_reads, dtype=np.int64)
    adata.layers["alt_fwd"] = _df_to_csc_sparse(alt_fwd_global, dtype=np.int64)
    adata.layers["alt_rev"] = _df_to_csc_sparse(alt_rev_global, dtype=np.int64)
    adata.layers["ref_fwd"] = _df_to_csc_sparse(ref_fwd_global, dtype=np.int64)
    adata.layers["ref_rev"] = _df_to_csc_sparse(ref_rev_global, dtype=np.int64)
    adata.layers["coverage"] = _df_to_csc_sparse(coverage_pos_global, dtype=np.int64)
    
    if verbose:
        print("Finished loading MGATK results.", flush=True)
    
    return adata


def LoadingMAEGATK_typewise(
    patient: str,
    samples_path: Path | None = None,
    barcodes_path: Path | None = None,
    samples_file: str | None = None,
    patient_column: str = "patient",
    reference_path: str | Path | None = None,
    chromosome_prefix: str = "chrM",
    type_use: str = "scRNAseq_MT",
    cells_include: set[str] | None = None,
    cells_exclude: set[str] | None = None,
    min_cells: int = 2,
    verbose: bool = True,
):
    """Memory‑efficient re‑implementation of the original mgatk loading code.
    
    Key optimisations
    • **Low‑width dtypes** – counts and positions use ``int32`` / ``uint32`` instead of the Pandas default ``int64``.
    • **Stay in Polars** for heavy joins/pivots; convert to Pandas only once per table.
    • **Sparse matrices** – all large 2‑D layers (``X`` and ``layers``) are stored as ``scipy.sparse.csc_matrix``.
    • **Float32 everywhere** for fractional layers and other floating‑point data.
    • *Aggressive* `del` + ``gc.collect`` helps release intermediate buffers promptly.
    """
    # 1. Here, we load the design matrix.
    if samples_path and barcodes_path:
        if verbose:
            print(f"Loading single sample {patient}.", flush=True)

        design_matrix = pd.DataFrame(
            {
                "patient": [patient],
                "sample": [patient],
                "input_path": [str(samples_path)],
                "cells": [str(barcodes_path)],
            }
        )
    else:
        if verbose:
            print(f"Loading samples for patient {patient}.", flush=True)
        
        design_matrix = pd.read_csv(samples_file, dtype=str)
        # if (patient_column not in design_matrix.columns) and (patient_column != "merge"):
        #     raise ValueError(f"Column '{patient_column}' not found in design matrix.")
        required_columns = {patient_column, "sample", "input_path", "cells", "source", "type"}
        missing = required_columns - set(design_matrix.columns)
        if missing:
            raise ValueError(f"The design matrix misses the following columns: {sorted(missing)}")
        
        design_matrix = design_matrix[design_matrix["source"].str.contains("maegatk", case=False, na=False)]
        if patient_column != "merge":
            design_matrix = design_matrix[design_matrix[patient_column] == patient]
        
        design_matrix = design_matrix[design_matrix["type"] == type_use]
    
    # Checking if we have any samples.
    if design_matrix.shape[0] == 0:
        raise ValueError("You have no samples in the design matrix. Please check.")
    
    # 2. Here, we get the reference genome positions once.
    if verbose:
        print("Reading the reference.", flush=True)
        print("Checking if the reference file exists.", flush=True)
    
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    ref_df = pd.read_csv(
        reference_path,
        sep="\t",
        header=None,
        names=["position", "ref_base"],
        dtype={"position": np.int32},
    )
    ref_df["ref_base"] = ref_df["ref_base"].str.upper()
    
    muts = [
        f"{chromosome_prefix}_{pos}_{ref}>{alt}"
        for pos, ref in zip(ref_df["position"], ref_df["ref_base"], strict=True)
        for alt in ("A", "C", "G", "T")
        if alt != ref
    ]
    # mut_positions = ref_df["position"].to_numpy(dtype = np.int32)
    mut_positions = [int(m.split("_")[1]) for m in muts]
    
    # Pre‑allocate global tables as INT32/FLOAT32.
    alt_fwd_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int32, 0))
    alt_fwd_qual_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.float32, 0.0))
    alt_rev_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int32, 0))
    alt_rev_qual_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.float32, 0.0))
    ref_fwd_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int32, 0))
    ref_fwd_qual_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.float32, 0.0))
    ref_rev_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int32, 0))
    ref_rev_qual_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.float32, 0.0))
    coverage_pos_global = pd.DataFrame(index=muts, dtype=pd.SparseDtype(np.int32, 0))
    
    # Polars copy for fast joins later
    ref_pl = pl.DataFrame(ref_df)
    
    # 3. Iterate over the samples
    # Schema to load the base file in polars.
    schema_common = {
        "position": pl.Int32,
        "cell": pl.Utf8,
        "fwd": pl.UInt32,
        "fwd_qual": pl.Float32,
        "rev": pl.UInt32,
        "rev_qual": pl.Float32,
    }
    # Schema to load the coverage file in polars.
    schema_cov = {"position": pl.Int32, "cell": pl.Utf8, "cov": pl.UInt32}
    
    for _, row in design_matrix.iterrows():
        # row = next(design_matrix.iterrows())[1]
        sample_use = row["sample"]
        folder = Path(row["input_path"])
        
        if verbose:
            print(f"Processing sample: {sample_use}.", flush=True)
        
        # 3a barcodes
        if verbose:
            print("Loading cell barcodes.", flush=True)
        
        cell_barcodes = (
            pl.read_csv(
                str(row["cells"]),
                has_header=False,
                separator="\t",
                new_columns=["cell"],
            )
            .select("cell")
            .to_series()
            .to_list()
        )
        
        if cells_include is not None:
            if verbose:
                print("Subsetting the barcodes to barcodes_include.", flush=True)
            
            cell_barcodes = [c for c in cell_barcodes if c in cells_include]
        
        if cells_exclude is not None:
            if verbose:
                print("Subsetting the barcodes to remove barcodes_exclude.", flush=True)
            
            cell_barcodes = [c for c in cell_barcodes if c not in cells_exclude]
        
        if not cell_barcodes:
            if verbose:
                print("No barcodes left after cell barcode filtering. Skipping sample {sample_use}.", flush=True)
            
            continue
        
        if verbose:
            print(
                f"We generate a list of all cell barcodes for {sample_use} to merge the different matrices.", flush=True
            )
        
        all_cell_barcodes = [f"{sample_use}_{c}" for c in cell_barcodes]
        
        # 3b – read base files
        if verbose:
            print("Reading A/C/G/T base files.", flush=True)
        
        dfs: list[pl.DataFrame] = []
        for base in ("A", "C", "G", "T"):
            # base = "A"
            path_use = folder / f"{sample_use}.{base}.txt.gz"
            if not os.path.isfile(path_use):
                path_use = folder / f"maegatk.{base}.txt.gz"
            
            if not os.path.isfile(path_use):
                raise FileNotFoundError(
                    f"Sample: {sample_use}. Neither the file {sample_use}.{base}.txt.gz nor the file maegatk.{base}.txt.gz exist. Please check."
                )
            
            if verbose:
                print(f"{path_use.name}", flush=True)
            
            dfs.append(
                pl.read_csv(
                    str(path_use),
                    has_header=False,
                    separator=",",
                    new_columns=["position", "cell", "fwd", "fwd_qual", "rev", "rev_qual"],
                    schema=schema_common,
                )
                .with_columns(pl.lit(base).alias("alt"))
                .filter(pl.col("cell").is_in(cell_barcodes))
            )
        
        all_bases = pl.concat(dfs)
        del dfs
        gc.collect()
        
        # We check if there are any duplications of variant/cell combinations.
        # We do this, since this should never happen. If it does, something is wrong.
        if verbose:
            print("We check if there are any duplications of variant/cell combinations.", flush=True)
        
        dup_all = all_bases.group_by(["position", "cell", "alt"]).len().filter(pl.col("len") > 1)
        if dup_all.height:
            print("Duplicated positions, cells, variants.", flush=True)
            print(dup_all.head(5), flush=True)
            raise ValueError("Duplicate (position, cell, alt) rows in the reads. Please check.")
        
        # 3c – ALT tables
        if verbose:
            print("Building ALT tables.", flush=True)
        
        alt_pl = (
            all_bases.join(ref_pl, on="position")
            .filter(pl.col("alt") != pl.col("ref_base"))
            .with_columns(
                pl.concat_str(
                    [
                        pl.lit(chromosome_prefix),
                        pl.lit("_"),
                        pl.col("position").cast(pl.Utf8),
                        pl.lit("_"),
                        pl.col("ref_base"),
                        pl.lit(">"),
                        pl.col("alt"),
                    ]
                ).alias("mutation")
            )
        )
        grp_alt = alt_pl.group_by(["mutation", "cell"]).agg(
            [
                pl.col("fwd").first().alias("fwd"),
                pl.col("fwd_qual").first().alias("fwd_qual"),
                pl.col("rev").first().alias("rev"),
                pl.col("rev_qual").first().alias("rev_qual"),
            ]
        )
        del alt_pl
        gc.collect()
        
        alt_fwd = grp_alt.pivot(values="fwd", index="mutation", on="cell", aggregate_function="first").fill_null(0)
        alt_fwd_qual = grp_alt.pivot(
            values="fwd_qual", index="mutation", on="cell", aggregate_function="first"
        ).fill_null(0)
        alt_rev = grp_alt.pivot(values="rev", index="mutation", on="cell", aggregate_function="first").fill_null(0)
        alt_rev_qual = grp_alt.pivot(
            values="rev_qual", index="mutation", on="cell", aggregate_function="first"
        ).fill_null(0)
        del grp_alt
        gc.collect()
        
        alt_fwd_pd = _pl_to_pd_int32(alt_fwd)
        alt_fwd_pd.columns = [f"{sample_use}_{c}" for c in alt_fwd_pd.columns]
        alt_fwd_pd = _align_cols(alt_fwd_pd, all_cell_barcodes, pd.SparseDtype(np.int32, 0))
        alt_fwd_pd = alt_fwd_pd.reindex(index=muts, fill_value=0)
        alt_fwd_global = (
            alt_fwd_global.join(alt_fwd_pd, how="outer").fillna(0).astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        
        alt_fwd_qual_pd = _pl_to_pd_float32(alt_fwd_qual)
        alt_fwd_qual_pd.columns = [f"{sample_use}_{c}" for c in alt_fwd_qual_pd.columns]
        alt_fwd_qual_pd = _align_cols(alt_fwd_qual_pd, all_cell_barcodes, pd.SparseDtype(np.float32, 0.0))
        alt_fwd_qual_pd = alt_fwd_qual_pd.reindex(index=muts, fill_value=0)
        alt_fwd_qual_global = (
            alt_fwd_qual_global.join(alt_fwd_qual_pd, how="outer")
            .fillna(0)
            .astype(pd.SparseDtype(np.float32, 0.0), copy=False)
        )
        
        alt_rev_pd = _pl_to_pd_int32(alt_rev)
        alt_rev_pd.columns = [f"{sample_use}_{c}" for c in alt_rev_pd.columns]
        alt_rev_pd = _align_cols(alt_rev_pd, all_cell_barcodes, pd.SparseDtype(np.int32, 0))
        alt_rev_pd = alt_rev_pd.reindex(index=muts, fill_value=0)
        alt_rev_global = (
            alt_rev_global.join(alt_rev_pd, how="outer").fillna(0).astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        
        alt_rev_qual_pd = _pl_to_pd_float32(alt_rev_qual)
        alt_rev_qual_pd.columns = [f"{sample_use}_{c}" for c in alt_rev_qual_pd.columns]
        alt_rev_qual_pd = _align_cols(alt_rev_qual_pd, all_cell_barcodes, pd.SparseDtype(np.float32, 0.0))
        alt_rev_qual_pd = alt_rev_qual_pd.reindex(index=muts, fill_value=0)
        alt_rev_qual_global = (
            alt_rev_qual_global.join(alt_rev_qual_pd, how="outer")
            .fillna(0)
            .astype(pd.SparseDtype(np.float32, 0.0), copy=False)
        )
        
        del alt_fwd_pd, alt_rev_pd, alt_fwd_qual_pd, alt_rev_qual_pd
        gc.collect()
        
        # 3d reference reads
        if verbose:
            print("Adding reference reads.", flush=True)
        
        ref_reads_pl = all_bases.join(ref_pl, on="position").filter(pl.col("alt") == pl.col("ref_base"))
        grp_ref = ref_reads_pl.group_by(["position", "cell"]).agg(
            [
                pl.col("fwd").first().alias("fwd"),
                pl.col("fwd_qual").first().alias("fwd_qual"),
                pl.col("rev").first().alias("rev"),
                pl.col("rev_qual").first().alias("rev_qual"),
            ]
        )
        del ref_reads_pl
        gc.collect()
        
        ref_fwd_pos = grp_ref.pivot(values="fwd", index="position", on="cell", aggregate_function="first").fill_null(0)
        ref_fwd_qual_pos = grp_ref.pivot(
            values="fwd_qual", index="position", on="cell", aggregate_function="first"
        ).fill_null(0)
        ref_rev_pos = grp_ref.pivot(values="rev", index="position", on="cell", aggregate_function="first").fill_null(0)
        ref_rev_qual_pos = grp_ref.pivot(
            values="rev_qual", index="position", on="cell", aggregate_function="first"
        ).fill_null(0)
        del grp_ref
        gc.collect()
        
        ref_fwd_pos_pd = _pl_pos_to_pd(ref_fwd_pos)
        ref_fwd_pos_pd.columns = [f"{sample_use}_{c}" for c in ref_fwd_pos_pd.columns]
        ref_fwd_pd = (
            ref_fwd_pos_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        ref_fwd_pd = _align_cols(ref_fwd_pd, all_cell_barcodes, pd.SparseDtype(np.int32, 0))
        
        ref_fwd_qual_pos_pd = _pl_pos_to_pd(ref_fwd_qual_pos)
        ref_fwd_qual_pos_pd.columns = [f"{sample_use}_{c}" for c in ref_fwd_qual_pos_pd.columns]
        ref_fwd_qual_pd = (
            ref_fwd_qual_pos_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(pd.SparseDtype(np.float32, 0.0), copy=False)
        )
        ref_fwd_qual_pd = _align_cols(ref_fwd_qual_pd, all_cell_barcodes, pd.SparseDtype(np.float32, 0.0))
        
        ref_rev_pos_pd = _pl_pos_to_pd(ref_rev_pos)
        ref_rev_pos_pd.columns = [f"{sample_use}_{c}" for c in ref_rev_pos_pd.columns]
        ref_rev_pd = (
            ref_rev_pos_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        ref_rev_pd = _align_cols(ref_rev_pd, all_cell_barcodes, pd.SparseDtype(np.int32, 0))
        
        ref_rev_qual_pos_pd = _pl_pos_to_pd(ref_rev_qual_pos)
        ref_rev_qual_pos_pd.columns = [f"{sample_use}_{c}" for c in ref_rev_qual_pos_pd.columns]
        ref_rev_qual_pd = (
            ref_rev_qual_pos_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(pd.SparseDtype(pd.SparseDtype(np.int32, 0), 0.0), copy=False)
        )
        ref_rev_qual_pd = _align_cols(ref_rev_qual_pd, all_cell_barcodes, pd.SparseDtype(np.float32, 0.0))
        
        ref_fwd_global = (
            ref_fwd_global.join(ref_fwd_pd, how="outer").fillna(0).astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        ref_fwd_qual_global = (
            ref_fwd_qual_global.join(ref_fwd_qual_pd, how="outer")
            .fillna(0)
            .astype(pd.SparseDtype(np.float32, 0.0), copy=False)
        )
        ref_rev_global = (
            ref_rev_global.join(ref_rev_pd, how="outer").fillna(0).astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        ref_rev_qual_global = (
            ref_rev_qual_global.join(ref_rev_qual_pd, how="outer")
            .fillna(0)
            .astype(pd.SparseDtype(np.float32, 0.0), copy=False)
        )
        
        del (
            ref_fwd_pd,
            ref_rev_pd,
            ref_fwd_pos_pd,
            ref_rev_pos_pd,
            ref_fwd_qual_pd,
            ref_rev_qual_pd,
            ref_fwd_qual_pos_pd,
            ref_rev_qual_pos_pd,
        )
        gc.collect()
        
        if verbose:
            print("Calculating the coverage.")
        
        # Here, we calculate the coverage using the reads in the base files.
        # cov_pl = (
        #    all_bases
        #    .with_columns((pl.col("fwd") + pl.col("rev")).alias("cov"))
        #    .group_by(["position", "cell"])
        #    .agg(pl.col("cov").sum().alias("cov"))
        # )
        # cov_samp = cov_pl.pivot(
        #    values = "cov", index = "position", on = "cell", aggregate_function = "first"
        # ).fill_null(0)
        # del cov_pl, all_bases
        # gc.collect()
        
        # cov_pd = _pl_pos_to_pd(cov_samp)
        # cov_pd.columns = [f"{sample_use}_{c}" for c in cov_pd.columns]
        # cov_pd = (
        #    cov_pd.reindex(index = mut_positions, fill_value = 0)
        #    .set_index(pd.Index(muts, name = "mutation"))
        #    .astype(np.int32, copy = False)
        # )
        # coverage_pos_global = (
        #    coverage_pos_global.join(cov_pd, how = "outer").fillna(0).astype(np.int32, copy = False)
        # )
        # del cov_pd, cov_samp
        # gc.collect()
        
        # Here, we use the coverage from the coverage file.
        path_use = folder / f"{sample_use}.coverage.txt.gz"
        if not os.path.isfile(path_use):
            path_use = folder / "maegatk.coverage.txt.gz"
        
        if not os.path.isfile(path_use):
            raise FileNotFoundError(
                f"Sample: {sample_use}. Neither the file {sample_use}.coverage.txt.gz nor the file maegatk.coverage.txt.gz exist. Please check."
            )
        
        cov_pl = pl.read_csv(
            str(path_use),
            has_header=False,
            separator=",",
            new_columns=["position", "cell", "cov"],
            schema=schema_cov,
        ).filter(pl.col("cell").is_in(cell_barcodes))
        cov_samp = cov_pl.pivot(values="cov", index="position", on="cell", aggregate_function="first").fill_null(0)
        del cov_pl
        gc.collect()
        
        cov_pd = _pl_pos_to_pd(cov_samp)
        cov_pd.columns = [f"{sample_use}_{c}" for c in cov_pd.columns]
        cov_pd = (
            cov_pd.reindex(index=mut_positions, fill_value=0)
            .set_index(pd.Index(muts, name="mutation"))
            .astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        cov_pd = _align_cols(cov_pd, all_cell_barcodes, pd.SparseDtype(np.int32, 0))
        
        coverage_pos_global = (
            coverage_pos_global.join(cov_pd, how="outer").fillna(0).astype(pd.SparseDtype(np.int32, 0), copy=False)
        )
        del cov_pd, cov_samp
        gc.collect()
    
    # 4. Post‑processing
    # The columns were not in the same order in the different data.frames.
    # We correct this here.
    
    if verbose:
        print("Aligning columns and rows.", flush=True)
    
    # We put the matrixes in a list. Then we can conveniently iterate over them.
    dfs = {
        "alt_fwd_global": alt_fwd_global,
        "alt_fwd_qual_global": alt_fwd_qual_global,
        "alt_rev_global": alt_rev_global,
        "alt_rev_qual_global": alt_rev_qual_global,
        "ref_fwd_global": ref_fwd_global,
        "ref_fwd_qual_global": ref_fwd_qual_global,
        "ref_rev_global": ref_rev_global,
        "ref_rev_qual_global": ref_rev_qual_global,
        "coverage_pos_global": coverage_pos_global,
    }
    # Using the alt_fwd_global dataframe as a reference.
    reference_name, reference_df = next(iter(dfs.items()))
    reference_rows = reference_df.index
    reference_cols = reference_df.columns
    
    # Build cell/variant unions (preserve reference order; append extras in encounter order)
    cells_union = reference_rows
    for df_use in dfs.values():
        cells_union = cells_union.union(df_use.index, sort=False)
    
    variant_union = reference_cols
    for df_use in dfs.values():
        variant_union = variant_union.union(df_use.columns, sort=False)
    
    # Decide expected dtype per frame (qualities are float32; counts/coverage are int32)
    _expected_dtype = {
        name: (pd.SparseDtype(np.float32, 0.0) if "qual" in name else pd.SparseDtype(np.int32, 0))
        for name in dfs.keys()
    }
    
    # Pad every matrix
    for name in list(dfs.keys()):
        dfs[name] = _pad_and_cast(dfs[name], cells_union, variant_union, _expected_dtype[name])
    
    # Integrity check (all shapes/labels identical now)
    _rows = cells_union
    _cols = variant_union
    for name, df_use in dfs.items():
        if not (df_use.index.equals(_rows) and df_use.columns.equals(_cols)):
            raise RuntimeError(f"Alignment failed for {name}.")
    
    # Now, we unpack the list again to use the reordered data.
    alt_fwd_global = dfs["alt_fwd_global"]
    alt_fwd_qual_global = dfs["alt_fwd_qual_global"]
    alt_rev_global = dfs["alt_rev_global"]
    alt_rev_qual_global = dfs["alt_rev_qual_global"]
    ref_fwd_global = dfs["ref_fwd_global"]
    ref_fwd_qual_global = dfs["ref_fwd_qual_global"]
    ref_rev_global = dfs["ref_rev_global"]
    ref_rev_qual_global = dfs["ref_rev_qual_global"]
    coverage_pos_global = dfs["coverage_pos_global"]
    del (
        dfs,
        reference_name,
        reference_df,
        reference_rows,
        reference_cols,
        df_use,
    )
    
    if verbose:
        print(f"Filtering variants with less than {min_cells} cells.", flush=True)
    
    # alt_reads = (alt_fwd_global + alt_rev_global).astype(np.int32, copy = False)
    # nonzero_cells = (alt_reads > 0).sum(axis = 1)
    # keep_mut = nonzero_cells >= min_cells
    S64 = pd.SparseDtype(np.int64, 0)
    S32 = pd.SparseDtype(np.int32, 0)
    alt_reads = alt_fwd_global.astype(S64) + alt_rev_global.astype(S64)
    nonzero_cells = alt_reads.astype(bool).sum(axis=1).astype(np.int32)
    keep_mut = nonzero_cells >= min_cells
    
    alt_reads = alt_reads.loc[keep_mut]
    alt_fwd_global = alt_fwd_global.loc[keep_mut]
    alt_fwd_qual_global = alt_fwd_qual_global.loc[keep_mut]
    alt_rev_global = alt_rev_global.loc[keep_mut]
    alt_rev_qual_global = alt_rev_qual_global.loc[keep_mut]
    ref_fwd_global = ref_fwd_global.loc[keep_mut]
    ref_fwd_qual_global = ref_fwd_qual_global.loc[keep_mut]
    ref_rev_global = ref_rev_global.loc[keep_mut]
    ref_rev_qual_global = ref_rev_qual_global.loc[keep_mut]
    coverage_pos_global = coverage_pos_global.loc[keep_mut]
    # ref_reads = (ref_fwd_global + ref_rev_global).astype(np.int32, copy = False)
    ref_reads = (ref_fwd_global.astype(S64) + ref_rev_global.astype(S64)).astype(S32)  # back to int32 if you want
    
    if alt_fwd_global.empty or alt_fwd_global.shape[0] == 0 or alt_fwd_global.shape[1] == 0:
        if verbose:
            print(
                f"Filtering left {alt_fwd_global.shape[0]} variants × {alt_fwd_global.shape[1]} cells; returning None.",
                flush=True,
            )
        
        return None
    
    if verbose:
        print("We calculate the mean coverage.", flush=True)
    
    coverage_pos_global = coverage_pos_global.astype(pd.SparseDtype(np.int32, 0), copy=False)
    row_sums = (
        coverage_pos_global.sum(axis=1)  # sparse row-reduction
        .astype(np.float64)  # stable division
        .to_numpy()
    )
    
    mean_cov_per_variant = pd.Series(
        row_sums / coverage_pos_global.shape[1],
        index=coverage_pos_global.index,
        dtype=np.float32,
    )
    
    if verbose:
        print("We calculate the fraction.", flush=True)
    
    fraction = (
        alt_reads.div(coverage_pos_global.where(coverage_pos_global.ne(0))).astype(np.float32, copy=False).fillna(0.0)
    )
    
    if verbose:
        print("We calculate the strand concordance.", flush=True)
    
    strand_concordance = alt_fwd_global.corrwith(alt_rev_global, axis=1).astype(np.float32)
    
    if verbose:
        print("We calculate the average quality.", flush=True)
    
    weighted_average_qualities = compute_weighted_qualities(
        alt_fwd=alt_fwd_global,
        alt_fwd_qual=alt_fwd_qual_global,
        alt_rev=alt_rev_global,
        alt_rev_qual=alt_rev_qual_global,
        ref_fwd=ref_fwd_global,
        ref_fwd_qual=ref_fwd_qual_global,
        ref_rev=ref_rev_global,
        ref_rev_qual=ref_rev_qual_global,
        alt_reads=alt_reads,
        ref_reads=ref_reads,
        check_alignment=False,  # We have already enforced alignment before. So, no need for this here.
    )
    
    if verbose:
        print("Creating AnnData object.", flush=True)
    
    obs = pd.DataFrame(index=alt_reads.columns)
    var = pd.DataFrame(index=alt_reads.index)
    var["strand_concordance"] = strand_concordance.loc[var.index]
    
    adata = ad.AnnData(X=_df_to_csc_sparse(alt_reads), obs=obs, var=var)
    adata.var["avg_coverage"] = mean_cov_per_variant.loc[adata.var_names]
    adata.var["quality"] = weighted_average_qualities["all_qual_per_variant"]
    adata.var["alt_qual_per_variant"] = weighted_average_qualities["alt_qual_per_variant"]
    adata.var["ref_qual_per_variant"] = weighted_average_qualities["ref_qual_per_variant"]
    
    adata.obs["alt_qual_per_cell"] = weighted_average_qualities["alt_qual_per_cell"]
    adata.obs["ref_qual_per_cell"] = weighted_average_qualities["ref_qual_per_cell"]
    adata.obs["quality"] = weighted_average_qualities["all_qual_per_cell"]
    
    # All layers sparse (float32)
    adata.layers["fraction"] = _df_to_csc_sparse(fraction)
    adata.layers["alt_reads"] = _df_to_csc_sparse(alt_reads)
    adata.layers["alt_reads_qual"] = _df_to_csc_sparse(weighted_average_qualities["alt_qual_global"])
    adata.layers["ref_reads"] = _df_to_csc_sparse(ref_reads)
    adata.layers["ref_reads_qual"] = _df_to_csc_sparse(weighted_average_qualities["ref_qual_global"])
    adata.layers["alt_fwd"] = _df_to_csc_sparse(alt_fwd_global)
    adata.layers["alt_fwd_qual"] = _df_to_csc_sparse(alt_fwd_qual_global)
    adata.layers["alt_rev"] = _df_to_csc_sparse(alt_rev_global)
    adata.layers["alt_rev_qual"] = _df_to_csc_sparse(alt_rev_qual_global)
    adata.layers["ref_fwd"] = _df_to_csc_sparse(ref_fwd_global)
    adata.layers["ref_fwd_qual"] = _df_to_csc_sparse(ref_fwd_qual_global)
    adata.layers["ref_rev"] = _df_to_csc_sparse(ref_rev_global)
    adata.layers["ref_rev_qual"] = _df_to_csc_sparse(ref_rev_qual_global)
    adata.layers["coverage"] = _df_to_csc_sparse(coverage_pos_global)
    adata.layers["quality"] = _df_to_csc_sparse(weighted_average_qualities["all_qual_global"])
    
    if verbose:
        print("Finished loading MAEGATK results.", flush=True)
    
    return adata
