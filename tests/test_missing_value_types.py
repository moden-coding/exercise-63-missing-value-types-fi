#!/usr/bin/env python3
"""Tests for the Missing Value Types pandas assignment."""

import unittest

import numpy as np

from src.missing_value_types import missing_value_types


class TestMissingValueTypes(unittest.TestCase):
    """missing_value_types() -> a DataFrame with a few missing values."""

    def test_shape(self):
        df = missing_value_types()
        self.assertEqual(
            df.shape,
            (6, 2),
            msg="missing_value_types() should return a DataFrame with shape "
            "(6, 2): six countries and two columns. Got shape %r."
            % (df.shape,),
        )

    def test_index(self):
        df = missing_value_types()
        np.testing.assert_array_equal(
            df.index,
            ["United Kingdom", "Finland", "USA", "Sweden", "Germany", "Russia"],
            err_msg="The DataFrame's index should be the six country names in "
            "this exact order.",
        )

    def test_columns(self):
        df = missing_value_types()
        np.testing.assert_array_equal(
            df.columns,
            ["Year of independence", "President"],
            err_msg="The DataFrame's columns should be 'Year of independence' "
            "and 'President', in this order.",
        )

    def test_dtypes(self):
        df = missing_value_types()
        self.assertEqual(
            df.dtypes[0],
            np.float64,
            msg="Column 0 ('Year of independence') should hold float64 "
            "values, since a missing year forces the column out of integer "
            "dtype. Got %r." % (df.dtypes[0],),
        )
        self.assertEqual(
            df.dtypes[1],
            object,
            msg="Column 1 ('President') should hold object (string) values. "
            "Got %r." % (df.dtypes[1],),
        )

    def test_nan(self):
        df = missing_value_types()
        m = df.isnull().values
        self.assertTrue(
            m[0, 0],
            msg="Expected a missing value at row 0 ('United Kingdom'), "
            "column 0 ('Year of independence').",
        )
        self.assertTrue(
            m[0, 1],
            msg="Expected a missing value at row 0 ('United Kingdom'), "
            "column 1 ('President').",
        )
        self.assertTrue(
            m[3, 1],
            msg="Expected a missing value at row 3 ('Sweden'), column 1 "
            "('President').",
        )
        self.assertTrue(
            m[4, 0],
            msg="Expected a missing value at row 4 ('Germany'), column 0 "
            "('Year of independence').",
        )
        s = m.sum()
        self.assertEqual(
            s,
            4,
            msg="Expected exactly 4 missing values total in the DataFrame, "
            "no more and no fewer. Got %d." % (s,),
        )


if __name__ == "__main__":
    unittest.main()
