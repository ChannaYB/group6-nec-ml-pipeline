"""
Unit tests for data ingestion module
"""

import pytest
import pandas as pd
import os
from src.data_ingestion import create_train_test
from src.config import TARGET_COLUMN, GROUP_COLUMN, NUM_PLANTS


def test_create_train_test_outputs():
    train_df, test_df = create_train_test(verbose=False)

    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0


def test_no_overlap_between_train_and_test_groups():
    train_df, test_df = create_train_test(verbose=False)

    # Unique group identifiers in each split
    train_groups = set(train_df[GROUP_COLUMN].unique())
    test_groups = set(test_df[GROUP_COLUMN].unique())

    # Demand IDs must not overlap
    assert train_groups.isdisjoint(test_groups), (
        "Demand ID leakage detected: some Demand IDs appear in both train and test"
    )


def test_each_group_has_expected_num_plants():
    train_df, test_df = create_train_test(verbose=False)

    # Count rows per demand in each split
    train_counts = train_df.groupby(GROUP_COLUMN).size()
    test_counts = test_df.groupby(GROUP_COLUMN).size()

    # All counts must equal the configured number of plants per demand
    assert (train_counts == NUM_PLANTS).all(), (
        f"Train split has Demand IDs with plant counts != {NUM_PLANTS}"
    )
    assert (test_counts == NUM_PLANTS).all(), (
        f"Test split has Demand IDs with plant counts != {NUM_PLANTS}"
    )


def test_target_has_no_missing_values():
    train_df, test_df = create_train_test(verbose=False)

    assert train_df[TARGET_COLUMN].isna().sum() == 0, (
        f"Train split contains missing values in target column: {TARGET_COLUMN}"
    )
    assert test_df[TARGET_COLUMN].isna().sum() == 0, (
        f"Test split contains missing values in target column: {TARGET_COLUMN}"
    )