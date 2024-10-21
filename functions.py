import scanpy as sc
import gzip
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import openslide
import numpy as np
from cv2 import perspectiveTransform


def parquet_to_csv(pth):
    spatial_path = os.path.join(pth, 'spatial')
    parquet_file = os.path.join(spatial_path, 'tissue_positions.parquet')
    csv_file = os.path.join(spatial_path, 'tissue_positions_list.csv')
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)
    
    if not os.path.exists(csv_file):
        # Write to CSV
        df.to_csv(csv_file, index=False)
        
        print(f"Converted {parquet_file} to {csv_file}")


def fix_xy(adata):
    x_fix = [float(x) for x in adata.obsm['spatial'][:,0]]
    y_fix = [float(y) for y in adata.obsm['spatial'][:,1]]
    xy_fix = np.array([x_fix, y_fix]).T
    adata.obsm['spatial'] = xy_fix
    return adata


def make_raw_matrix(pth, adata):
    # Load barcodes (spot identifiers)
    barcodes_file = os.path.join(pth, "filtered_feature_bc_matrix/barcodes.tsv.gz")

    # Since the barcodes are gzipped, we'll read them using gzip
    with gzip.open(barcodes_file, 'rt') as f:
        barcodes = [line.strip() for line in f.readlines()]

        # Load gene names (features)
    features_file = os.path.join(pth, "filtered_feature_bc_matrix/features.tsv.gz")
    with gzip.open(features_file, 'rt') as f:
        features = [line.strip().split("\t")[1] for line in f.readlines()]

    # get raw data matrix
    dat = adata.X.toarray()

    # Convert raw matrix to a DataFrame and add barcodes and gene names
    raw_matrix_df = pd.DataFrame(dat, index=barcodes, columns=features)

    return raw_matrix_df


def qc_spots_and_norm(adata, min_genes, min_cells, filter_mt=True, highly_var=True, target_sum=1e4):
    # QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # if mitochondrial genes are labeled with 'MT-'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter cells based on QC metrics (adjust thresholds based on your data)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if filter_mt:
        adata = adata[adata.obs.pct_counts_mt < 5, :]

    # Normalize 
    # Normalize each cell by total counts, scaling to 10,000 reads per cell by default, and log-transform
    sc.pp.normalize_total(adata, target_sum)
    sc.pp.log1p(adata)

    if highly_var:
        print('selecting highly variable genes')
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        # Subset to highly variable genes
        adata = adata[:, adata.var['highly_variable']]
    return adata


# https://github.com/10XGenomics/janesick_nature_comms_2023_companion/blob/main/companion_functions.py
def transform_coordinates(
    coords: np.ndarray[np.float32, (None, 2)], transform_matrix: np.ndarray[np.float32]
) -> np.ndarray[np.float32, (None, 2)]:
    """Transforms coordinates using transform.
    https://en.wikipedia.org/wiki/Homography_(computer_vision)
    """
    return perspectiveTransform(
        coords.reshape(-1, 1, 2).astype(np.float32), transform_matrix
    )[:, 0, :]

def make_full_res_xy(adata, spatial_pth, aff, ds, buffer_size):
    # align to high rez and then apply tranformation to downsized npdi
    tissue_positions_pth = os.path.join(spatial_pth, 'tissue_positions.parquet')
    if not os.path.exists(tissue_positions_pth):
        tissue_positions_pth = os.path.join(spatial_pth, 'tissue_positions.csv')
        # read csv
    
    # read parquet if exists
    tissue_positions_df = pd.read_parquet(tissue_positions_pth)
    tissue_positions_df = tissue_positions_df.loc[tissue_positions_df['in_tissue']==1]

    scale_factors_pth = os.path.join(spatial_pth, 'scalefactors_json.json')
    with open(scale_factors_pth, 'rb') as f:
        scale_factors = json.load(f)

    adata = fix_xy(adata)

    xy = adata.obsm['spatial']*scale_factors['tissue_hires_scalef']
    xy_reg = transform_coordinates(xy, aff)
    xy_ndpi = xy_reg*ds

    min_x = min(xy_ndpi[:,0])-buffer_size
    max_x = max(xy_ndpi[:,0])+buffer_size
    min_y = min(xy_ndpi[:,1])-buffer_size
    max_y = max(xy_ndpi[:,1])+buffer_size

    adata.obsm['spatial'] = xy_ndpi
    return adata, [min_x, max_x, min_y, max_y]


def read_ndpi(ndpi_pth, npdi_crop_coords):

    [min_x, max_x, min_y, max_y] = npdi_crop_coords

    # Open the .ndpi file
    slide = openslide.OpenSlide(ndpi_pth)

    # Print the full resolution dimensions for debugging
    full_width, full_height = slide.dimensions
    print(f"Full resolution dimensions: {full_width} x {full_height}")

    # Read the cropped region at level 0 (highest resolution)
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    cropped_region = slide.read_region((int(min_x), int(min_y)), 0, (width, height))

    # Convert the image to RGB (PIL Image)
    img = cropped_region.convert("RGB")
    return img


def read_ndpi_pix_ajd_fix(ndpi_pth, npdi_crop_coords,pix_ratio):

    [min_x, max_x, min_y, max_y] = npdi_crop_coords

    # Open the .ndpi file
    slide = openslide.OpenSlide(ndpi_pth)

    # Print the full resolution dimensions for debugging
    full_width, full_height = slide.dimensions
    print(f"Full resolution dimensions: {full_width} x {full_height}")

    # Read the cropped region at level 0 (highest resolution)
    width = int(max_x - min_x)
    height = int(max_y - min_y)

    # resizing from 0.45 um/pix to 0.5 causes it to loose the edge, add back here
    # pix_ratio = 0.5 - 0.4416 (example)
    width = width + int(pix_ratio*width)
    height = height + int(pix_ratio*width)

    cropped_region = slide.read_region((int(min_x), int(min_y)), 0, (width, height))

    # Convert the image to RGB (PIL Image)
    img = cropped_region.convert("RGB")
    return img
