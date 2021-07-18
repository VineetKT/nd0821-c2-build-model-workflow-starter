import logging

import numpy as np
import pandas as pd
import scipy.stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_column_names(data):
    """Test if column names in the input data match"""

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns), \
        logger.info("Column names in the input data doesn't match")


def test_neighborhood_names(data):
    """Test if Neighborhood names in the input data match"""

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh), \
        logger.info("Neighborhood names in the input data doesn't match")


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, - 73.50) & \
        data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0, \
        logger.info("Column names in the input data doesn't match")


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    """To validate if the data has reasonable size."""

    row_count = data.shape[0]

    assert 15000 < row_count < 100000, \
        logger.info("The dataset size is not sufficent or more than needed.")


def test_price_range(data, min_price, max_price):
    """To check if price values lies within reasonable range."""

    idx = data['price'].between(min_price, max_price)

    assert np.sum(~idx) == 0, \
        logger.info("Price column has outliers...")
