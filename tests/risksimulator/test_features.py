"""Unit testing the feature module."""
import numpy as np
import pandas as pd
import pytest
from kundenscore.datasetconstructor import features


class TestFeature:
    """Meta class for Test suites for single features."""

    ORDER_LEVEL_INDEX_COLS = ["sample_id", "order_id"]
    SAMPLE_LEVEL_INDEX_COLUMNS = ["sample_id"]
    _EXAMPLE_DATA_CACHE = None

    @pytest.fixture(scope="module")
    def example_data(self):
        """A fixture to provide example sales data to test the features on."""
        if not self._EXAMPLE_DATA_CACHE:
            n_samples = 50
            n_customers = 50
            distribution_channels = ["DE", "CH", "AT", "NL"]

            # some sample customers
            customers = pd.DataFrame()
            customers["customer_id"] = np.arange(n_customers) + 1
            customers["customer_number"] = customers.customer_id * 3 - 1
            customers["distribution_channel"] = np.random.choice(
                distribution_channels, size=n_customers
            )
            customers["acquisition_date"] = pd.date_range(
                start="1980-01-01", end="2020-01-01", periods=n_customers
            )
            customers["gender"] = np.random.choice(
                ["männlich", "weiblich", "divers"], size=n_customers
            )
            delta = np.random.randint(18 * 365, 100 * 365, size=n_customers)
            customers["birthday"] = customers.acquisition_date - delta * pd.Timedelta(
                "1d"
            )

            data = list()
            min_pred_time = pd.Timestamp("2016-01-01")
            sample_id = 1
            min_order_id = 1
            min_orderposition_id = 1
            for sample_id in range(n_samples):
                sample_data = pd.Series()
                sample_data["sample_id"] = sample_id
                sample_data = sample_data.append(customers.sample(n=1).squeeze())

                delta = np.random.uniform()
                sample_data["prediction_time"] = min_pred_time + delta * pd.Timedelta(
                    "4y"
                )

                n_orders = np.random.randint(1, 10)
                for order_id in range(min_order_id, min_order_id + n_orders):

                    order_dates = pd.date_range(
                        "2016-01-01", sample_data.prediction_time, freq="1d"
                    )
                    order_data = sample_data.copy()
                    order_data["order_id"] = order_id
                    order_data["order_date"] = np.random.choice(order_dates)
                    order_data["distribution_channel"] = np.random.choice(
                        distribution_channels
                    )

                    n_positions = np.random.randint(1, 6)
                    for orderposition_id in range(
                        min_orderposition_id, min_orderposition_id + n_positions
                    ):
                        position_data = order_data.copy()
                        position_data["orderposition_id"] = orderposition_id

                        demand_value = np.random.uniform(0, 100)
                        demand_qty = int(demand_value / np.random.uniform(1, 100))
                        position_data[
                            "demand_without_cancellations_value"
                        ] = demand_value
                        position_data["demand_without_cancellations_qty"] = demand_qty
                        position_data[
                            "net_turnover_value"
                        ] = demand_value * np.random.uniform(0.5, 1)
                        position_data[
                            "net_turnover_qty"
                        ] = demand_qty * np.random.uniform(0.5, 1)
                        shipping_date = order_data.order_date + np.random.uniform() * pd.Timedelta(
                            "6w"
                        )
                        position_data["shipping_date"] = shipping_date
                        data.append(position_data)
                    min_orderposition_id = orderposition_id
                min_order_id = order_id
            self._EXAMPLE_DATA_CACHE = pd.DataFrame(data)

        return self._EXAMPLE_DATA_CACHE.copy()

    @pytest.fixture
    def index_data(self):
        """Example data with all index columns for ODSFeatures."""
        data = pd.DataFrame()
        data["sample_id"] = [1, 1, 1, 1, 2, 2]
        data["order_id"] = [1, 1, 2, 2, 3, 4]
        return data


class TestSampleLevelFeature(TestFeature):
    """Test suite for kundenscore.datasetconstructor.features.SampleLevelFeature."""

    AGG_COLS = [
        "customer_id",
        "customer_number",
        "distribution_channel",
        "gender",
        "prediction_time",
    ]

    @pytest.mark.parametrize(
        "missing_col",
        features.SampleLevelFeature.AGG_COLS
        + [features.SampleLevelFeature.SAMPLE_LEVEL_INDEX_COLUMNS],
    )
    def test_transform_w_missing_columns_fails(self, example_data, missing_col):
        """
        GIVEN example data with a missing column and a SampleLevelFeature
        WHEN the transform method of the feature is applied to the data
        THEN an exception is raised.
        """
        example_data = example_data.drop(missing_col, axis=1)
        feature = features.SampleLevelFeature()
        with pytest.raises(Exception):
            feature.transform(example_data)

    def test_transform(self, example_data):
        """
        GIVEN a sample level feature and some example data
        WHEN the transform method of the feature is applied to the data
        THEN the resulting data equals the result from applying a pandas groupby first
            with column selection to the data.
        """
        feature = features.SampleLevelFeature()
        result = feature.transform(example_data)
        expected = example_data.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)[
            self.AGG_COLS
        ].first()
        pd.testing.assert_frame_equal(result, expected)

    def test_instantiation(self):
        """
        WHEN a SampleLevelFeature is instantiated without parameters
        THEN it works smoothly.
        """
        features.SampleLevelFeature()


class TestAverageOrderValue(TestFeature):
    """Test suite for kundenscore.datasetconstructor.features.AverageOrderValue."""

    AGGREGATION_COLUMN = "net_turnover_value"
    FEATURE_NAME = "aov"

    def test_instantiation(self):
        """
        WHEN a AverageOrderValue-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.AverageOrderValue()

    def test_transform(self, example_data):
        """
        GIVEN a AverageOrderLevel-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the total net turnover value of the positions in the sample.
        """
        feature = features.AverageOrderValue()
        result = feature.transform(example_data)
        order_value = example_data.groupby(self.ORDER_LEVEL_INDEX_COLS)[
            self.AGGREGATION_COLUMN
        ].sum()
        expected = (
            order_value.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)
            .mean()
            .rename(self.FEATURE_NAME)
        )
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


class TestTotalOrderValue(TestFeature):
    AGGREGATION_COLUMN = "net_turnover_value"
    FEATURE_NAME = "total_order_value"

    def test_instantiation(self):
        """
        WHEN a TotalOrderValue-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.TotalOrderValue()

    def test_transform(self, example_data):
        """
        GIVEN a TotalOrderLevel-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the total net turnover value of the positions in the sample.
        """
        feature = features.TotalOrderValue()
        result = feature.transform(example_data)
        expected = (
            example_data.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)[
                self.AGGREGATION_COLUMN
            ]
            .sum()
            .rename(self.FEATURE_NAME)
        )
        pd.testing.assert_series_equal(result, expected)


class TestAverageOrderQuantity(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.AverageOrderQuantity."""

    AGGREGATION_COLUMN = "net_turnover_qty"
    FEATURE_NAME = "aoq"

    def test_instantiation(self):
        """
        WHEN a AverageOrderQuantity-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.AverageOrderQuantity()

    def test_transform(self, example_data):
        """
        GIVEN a AverageOrderQuantity-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the total net turnover quantity for all positions
            in the sample.
        """
        feature = features.AverageOrderQuantity()
        result = feature.transform(example_data)

        order_quantity = example_data.groupby(self.ORDER_LEVEL_INDEX_COLS)[
            self.AGGREGATION_COLUMN
        ].sum()
        expected = (
            order_quantity.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)
            .mean()
            .rename(self.FEATURE_NAME)
        )

        pd.testing.assert_series_equal(result, expected)


class TestMaxShippingDelayLastOrder(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.MaxShippingDelayLastOrder."""

    def test_instantiation(self):
        """
        WHEN a MaxShippingDelayLastOrder-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.MaxShippingDelayLastOrder()

    def test_transform(self, example_data):
        """
        GIVEN a MaxShippingDelayLastOrder-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the maximum shipping delay for the last order of each
            ample's  customer.
        """
        feature = features.MaxShippingDelayLastOrder()
        result = feature.transform(example_data)

        # delays for all order positions
        example_data["delay"] = example_data.shipping_date - example_data.order_date
        example_data["delay"] = example_data.delay / pd.Timedelta("1d")
        # maximum delay per order
        max_delays = example_data.groupby(self.ORDER_LEVEL_INDEX_COLS)[
            "delay", "order_date"
        ].agg({"delay": "max", "order_date": "first"})
        # sort by order date ascencing
        max_delays = max_delays.sort_values(
            ["sample_id", "order_date"], ascending=False
        )
        # the first occurence per sample is the maximally
        # delayed item from the last order
        expected = (
            max_delays.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)
            .delay.first()
            .rename("max_shipping_delay_last_order")
        )

        pd.testing.assert_series_equal(result, expected)


class TestAgeAtPredictionTime(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.AgeAtPredictionTime."""

    def test_instantiation(self):
        """
        WHEN a AgeAtPredictionTime-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.AgeAtPredictionTime()

    def test_transform(self, example_data):
        """
        GIVEN a AgeAtPredictionTime-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the age of a sample's customer at prediction time in years.
        """
        feature = features.AgeAtPredictionTime()
        result = feature.transform(example_data)

        example_data["customer_age"] = (
            example_data.prediction_time - example_data.birthday
        )
        example_data["customer_age"] = example_data.customer_age / pd.Timedelta("1y")
        expected = example_data.groupby("sample_id").customer_age.first()

        pd.testing.assert_series_equal(result, expected)


class TestAverageOrderFrequency(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.AverageOrderFrequency."""

    def test_instantiation(self):
        """
        WHEN a AverageOrderFrequency-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.AverageOrderFrequency()

    def test_transform(self, example_data):
        """
        GIVEN a AverageOrderFrequency-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the average order per months in the lookback of the sample.
        """
        feature = features.AverageOrderFrequency()
        result = feature.transform(example_data)

        lookback = pd.to_timedelta("2y")

        lifetime_covered = example_data.prediction_time - example_data.acquisition_date
        lifetime_covered = lifetime_covered.clip(upper=lookback)
        lifetime_covered = lifetime_covered / pd.to_timedelta("30.5d")
        lifetime_covered = lifetime_covered.groupby(example_data.sample_id).first()

        n_orders = example_data.groupby("sample_id").order_id.nunique()

        expected = n_orders / lifetime_covered
        expected.name = "orders_per_month"

        pd.testing.assert_series_equal(result, expected)


class TestPredictionMonth(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.PredictionMonth."""

    def test_instantiation(self):
        """
        WHEN a PredictionMonth-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.PredictionMonth()

    def test_transform(self):
        """
        GIVEN a PredictionMonth-Feature and some example data
        WHEN the feature is applied to the data
        THEN the result is the lowercase name of the month of the prediction time.
        """
        input_data = pd.DataFrame(
            {
                "prediction_time": [
                    pd.to_datetime("2018-01-12"),
                    pd.to_datetime("2018-01-12"),
                    pd.to_datetime("2019-05-03"),
                    pd.to_datetime("1980-03-01"),
                ],
                "sample_id": [1, 1, 2, 3],
            }
        )
        expected = pd.Series(
            ["january", "may", "march"],
            index=pd.Index([1, 2, 3], name="sample_id"),
            name="prediction_month",
        )

        feature = features.PredictionMonth()
        result = feature.transform(input_data)

        pd.testing.assert_series_equal(result, expected)


class TestOrderValueBinned(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.OrderValueBinned."""

    def test_instantiation(self):
        """
        WHEN a OrderDemandBinned-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.OrderDemandBinned()

    def test_transform(self, example_data):
        """
        GIVEN a OrderValueBinned-Feature and some example data
        WHEN the feature is applied to the data
        THEN the demand of the sample's lookback binned by half year bins.
        """
        recency = example_data.prediction_time - example_data.order_date

        bin_size = pd.to_timedelta("0.5y")
        bins = np.ceil(recency / bin_size).astype(int)
        bin_size_days = int(bin_size / pd.to_timedelta("1d"))
        bins = bins.apply(lambda b: f"bin_{b*bin_size_days}d")
        example_data["bin"] = bins

        order_value_by_bin = pd.pivot_table(
            example_data,
            values=["demand_without_cancellations_value"],
            index="sample_id",
            columns="bin",
            aggfunc="sum",
            fill_value=0,
        )
        order_value_by_bin = features.ODSFeature.flatten_and_format_column_index(
            order_value_by_bin
        )
        order_value_by_bin_scaled = order_value_by_bin.div(
            order_value_by_bin.sum(axis=1), axis=0
        )
        expected = order_value_by_bin_scaled.add_suffix("_scaled")

        feature = features.OrderDemandBinned()
        result = feature.transform(example_data)

        pd.testing.assert_frame_equal(result, expected)


class TestOrderQuantityBinned(TestFeature):
    """Tests for kundenscore.datasetconstructor.features.OrderQuantityBinned."""

    def test_instantiation(self):
        """
        WHEN a OrderQuantityBinned-Feature is instantiated without parameters
        THEN it works smoothly.
        """
        features.OrderQuantityBinned()

    def test_transform(self, example_data):
        """
        GIVEN a OrderQuantityBinned-Feature and some example data
        WHEN the feature is applied to the data
        THEN the demand quantity of the sample's lookback binned by half year bins.
        """
        recency = example_data.prediction_time - example_data.order_date

        bin_size = pd.to_timedelta("0.5y")
        bins = np.ceil(recency / bin_size).astype(int)
        bin_size_days = int(bin_size / pd.to_timedelta("1d"))
        bins = bins.apply(lambda b: f"bin_{b*bin_size_days}d")
        example_data["bin"] = bins

        order_value_by_bin = pd.pivot_table(
            example_data,
            values=["demand_without_cancellations_qty"],
            index="sample_id",
            columns="bin",
            aggfunc="sum",
            fill_value=0,
        )
        order_value_by_bin = features.ODSFeature.flatten_and_format_column_index(
            order_value_by_bin
        )
        order_value_by_bin_scaled = order_value_by_bin.div(
            order_value_by_bin.sum(axis=1), axis=0
        )
        expected = order_value_by_bin_scaled.add_suffix("_scaled")

        feature = features.OrderQuantityBinned()
        result = feature.transform(example_data)

        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "by,agg_col,agg_method",
    [
        ("sample_id", "net_turnover_value", "sum"),
        (["sample_id", "order_id"], "net_turnover_value", "mean"),
        ("sample_id", "customer_id", "first"),
        ("sample_id", "order_date", "max"),
    ],
)
class TestGroupByAggregateFeature(TestFeature):
    def test_to_dict(self, by, agg_col, agg_method):
        """
        GIVEN a set of parameters
        WHEN a GroupByAggregate-Feature is instatiated with them and its to_dict
            method is invoked
        THEN the result contains the parameters.
        """
        feature = features.GroupByAggregateFeature(
            by=by, agg_col=agg_col, agg_method=agg_method
        )
        result = feature.to_dict()

        expected = dict(by=by, agg_col=agg_col, agg_method=agg_method)
        assert result == expected

    def test_instantiation(self, by, agg_col, agg_method):
        """
        WHEN a AverageOrderValue-Feature is instantiated with valid parameters
        THEN it works smoothly.
        """
        features.GroupByAggregateFeature(by=by, agg_col=agg_col, agg_method=agg_method)

    def test_transform(self, by, agg_col, agg_method, example_data):
        """
        GIVEN a GroupByAggregate-Feature and some example data
        WHEN the feature is applied to the data
        THEN the data is aggregated and grouped accordingly.
        """
        feature = features.GroupByAggregateFeature(
            by=by, agg_col=agg_col, agg_method=agg_method
        )
        result = feature.transform(example_data)

        expected = example_data.groupby(by)[agg_col].agg(agg_method)

        pd.testing.assert_series_equal(result, expected)


class TestODSFeature(TestFeature):
    """Unit testing the ODSFeature class."""

    @pytest.fixture
    def pivot_test_data(self):
        """Generate pivot example data."""
        pivot_data = pd.DataFrame(None)
        pivot_data["pivot"] = ["pivot1", "pivot1", "pivot1", "pivot2", "pivot2"]
        pivot_data["value_column"] = [2, 3, 6, 4, 6]
        pivot_data["ind"] = ["ind1", "ind1", "ind2", "ind1", "ind1"]
        return pivot_data

    @pytest.fixture
    def pivot_expected_result_data(self):
        """
        Returns the expected result without suffixes or prefixes on the
            'pivot_test_data'.
        """
        expected_frame = pd.DataFrame(None)
        expected_frame["pivot1"] = [1 / 3, 1.0]
        expected_frame["pivot2"] = [2 / 3, 0.0]
        expected_frame["ind"] = ["ind1", "ind2"]
        expected_frame = expected_frame.set_index("ind")
        return expected_frame

    @pytest.fixture
    def pivot_configuration(self):
        """Instantiate a pivot feature."""
        pivot_config = dict()
        pivot_config["columns"] = "pivot_column"
        pivot_config["aggfunc"] = "max"
        pivot_config["fill_value"] = 10
        pivot_config["values"] = "value_column"
        pivot_config["index"] = ["index1", "index2"]
        pivot_config["prefix"] = "pre"
        pivot_config["suffix"] = "suf"
        return pivot_config

    def test_transform_pivot_prefix(self, pivot_test_data, pivot_expected_result_data):
        """
        GIVEN the pivot test dataframe and the expected result
        WHEN the n'transform' method is used
        THEN the values are the same.
        """
        result_frame = features.ODSFeature().pivot_data(
            pivot_test_data,
            columns=["pivot"],
            aggfunc="sum",
            fill_value=0,
            values="value_column",
            prefix="piv_",
            index="ind",
        )

        expected_result = pivot_expected_result_data
        expected_result.columns = ["piv_pivot1", "piv_pivot2"]
        pd.testing.assert_frame_equal(result_frame, expected_result)

    def test_transform_pivot_suffix(self, pivot_test_data, pivot_expected_result_data):
        """
        GIVEN the pivot test data frame and an expected outcome with the suffix 'suf'.
        WHEN a ODSFeature is instantiated and the pivot test data is transformed
        THEN the values of the result frame are equal to the values of the expected
            frame.
        """
        result_frame = features.ODSFeature().pivot_data(
            pivot_test_data,
            columns="pivot",
            aggfunc="max",
            fill_value=0,
            values="value_column",
            suffix="_suf",
            index="ind",
        )

        expected_result = pivot_expected_result_data
        expected_result.columns = ["pivot1_suf", "pivot2_suf"]
        pd.testing.assert_frame_equal(result_frame, expected_result)

    def test_pivot_transform_zero_nan_division(self):
        """
        GIVEN pivot data with zero and NaN values.
        WHEN a ODSFeature is instantiated and the 'transform' method is applied on the
            created pivot data.
        THEN the returned values are equal to the expected values.
        """
        pivot_data = pd.DataFrame()
        pivot_data["pivot"] = [
            "pivot1",
            "pivot1",
            "pivot2",
            "pivot2",
            "pivot3",
            "pivot3",
        ]
        pivot_data["value_column"] = [0, 0, 0, np.nan, np.nan, np.nan]
        pivot_data["dummy"] = len(pivot_data["pivot"]) * ["dummy"]
        pivot_data["ind"] = len(pivot_data["pivot"]) * ["ind"]

        result_frame = features.ODSFeature().pivot_data(
            pivot_data,
            columns="pivot",
            aggfunc="sum",
            fill_value=0,
            values="value_column",
            index="ind",
        )

        expected_result = pd.DataFrame()
        expected_result["pivot1"] = [np.nan]
        expected_result["pivot2"] = [np.nan]
        expected_result["pivot3"] = [np.nan]
        expected_result["ind"] = ["ind"]
        expected_result = expected_result.set_index("ind")

        pd.testing.assert_frame_equal(result_frame, expected_result)


class TestDemandByProductCategoryScaled(TestFeature):
    """Unit testing DemandByProductCategoryScaled."""

    def test_demandbyproductcategoryscaled_index_data(self, index_data):
        """
        GIVEN some index data
        WHEN the necessary columns are added to the index data and the feature
            `DemandByProductCategoryScaled` is calculated
        THEN the transformed data is the same as the expected outcome.
        """
        index_data["product_category"] = [
            "Schuhe",
            "Haushalt",
            "Putzen",
            "Putzen",
            "Haushalt",
            "Putzen",
        ]
        index_data["demand_without_cancellations_value"] = [
            10.0,
            20.0,
            5.0,
            5.0,
            0.0,
            1.0,
        ]
        result_frame = features.DemandByProductCategoryScaled().transform(index_data)

        expected_frame = pd.DataFrame()
        expected_frame["sample_id"] = [1, 2]
        expected_frame = expected_frame.set_index("sample_id")
        expected_frame["demand_by_product_category_haushalt"] = [0.5, 0.0]
        expected_frame["demand_by_product_category_putzen"] = [0.25, 1.0]
        expected_frame["demand_by_product_category_schuhe"] = [0.25, 0.0]

        pd.testing.assert_frame_equal(result_frame, expected_frame)


class TestDemandByOrderChannelScaled(TestFeature):
    """Unit testing the class DemandByOrderChannelScaled."""

    def test_demandbyorderchannelscaled_index_data(self, index_data):
        """
        GIVEN some index data.
        WHEN the necessary columns are added to the index data and the
            feature `DemandByOrderChannelScaled` is calculated
        THEN the expected results are equal to the transformed index data.
        """
        index_data["order_channel"] = [
            "Internet",
            "Elektronisch",
            "E-Mail",
            "Telefon",
            "Telefon",
            "Beleg",
        ]
        index_data["demand_without_cancellations_value"] = [
            10.0,
            5.0,
            5.0,
            20.0,
            0.0,
            1.0,
        ]
        result_frame = features.DemandByOrderChannelScaled().transform(index_data)

        expected_frame = pd.DataFrame()
        expected_frame["sample_id"] = [1, 2]
        expected_frame = expected_frame.set_index("sample_id")
        expected_frame["demand_by_order_channel_scaled_online"] = [0.5, 0.0]
        expected_frame["demand_by_order_channel_scaled_offline"] = [0.5, 1.0]

        pd.testing.assert_frame_equal(
            result_frame.sort_index(axis=1), expected_frame.sort_index(axis=1)
        )


class TestDemandByPaymentChannelScaled(TestFeature):
    """Unit testing the class DemandByPaymentChannelScaled."""

    def test_demandbypaymentchannelscaled(self, index_data):
        """
        GIVEN some index data
        WHEN the necessary columns are added to the
            index data and the feature `DemandByPaymentChannelScaled` is calculated
        THEN the transformed index data equals the expected dataframe.
        """
        index_data["payment_channel"] = [
            "Karte",
            "Karte",
            "Bank",
            "Bank",
            "Rechnung",
            "Rechnung",
        ]
        index_data["demand_without_cancellations_value"] = [
            10.0,
            5.0,
            0.0,
            5.0,
            1.0,
            0.0,
        ]
        result_frame = features.DemandByPaymentChannelScaled().transform(index_data)

        expected_frame = pd.DataFrame()
        expected_frame["sample_id"] = [1, 2]
        expected_frame = expected_frame.set_index("sample_id")
        expected_frame["demand_by_payment_channel_scaled_karte"] = [0.75, 0.0]
        expected_frame["demand_by_payment_channel_scaled_bank"] = [0.25, 0.0]
        expected_frame["demand_by_payment_channel_scaled_rechnung"] = [0.0, 1.0]

        pd.testing.assert_frame_equal(
            result_frame.sort_index(axis=1), expected_frame.sort_index(axis=1)
        )


class TestRecency(TestFeature):
    """Unit testing the class recency."""

    def test_recency_index_data(self, index_data):
        """
        GIVEN the index dataframe.
        WHEN the class `Recency` is instantiated and the data is transformed
        THEN the result data is the same then the expected data outcome.
        """
        index_data["order_date"] = [
            "2019-01-01",
            "2019-01-01",
            "2019-01-02",
            "2019-01-02",
            "2019-12-31",
            "2020-01-01",
        ]
        index_data["prediction_time"] = [
            "2020-01-04",
            "2020-01-04",
            "2020-01-04",
            "2020-01-04",
            "2020-01-01",
            "2020-01-01",
        ]
        index_data["order_date"] = pd.to_datetime(index_data["order_date"])
        index_data["prediction_time"] = pd.to_datetime(index_data["prediction_time"])

        result_series = features.Recency().transform(index_data)
        index_series = pd.Series([1, 2], name="sample_id")
        expected_series = pd.Series([367.0, 0.0], name="recency", index=index_series)

        pd.testing.assert_series_equal(result_series, expected_series)

    def test_recency_example_data(self, example_data):
        """
        GIVEN example data
        WHEN the class recency is instantiated and the method 'transform' is used
        THEN the result is the same than if the logic of transform is rebuilt.
        """
        feature = features.Recency()
        result_frame = feature.transform(example_data)

        expected_frame = (
            example_data.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)
            .agg({"order_date": "max", "prediction_time": "first"})
            .pipe(
                features.ODSFeature.calculate_time_difference,
                ref_date="order_date",
                date_column="prediction_time",
                unit="1d",
            )
            .rename("recency")
        )

        pd.testing.assert_series_equal(result_frame, expected_frame)


class TestCustomerLifetime(TestFeature):
    """Unit testing the class CustomerLifetime."""

    def test_customerlifetime_index_data(self, index_data):
        """
        GIVEN some index data.
        WHEN the CustomerLifetime feature is calculated
        THEN the transformed data equals the expected outcome.
        """
        index_data["acquisition_date"] = [
            "1999-01-01",
            "1999-01-01",
            "1999-01-01",
            "1999-01-01",
            "2020-01-01",
            "2020-01-01",
        ]
        index_data["prediction_time"] = [
            "2020-01-04",
            "2020-01-04",
            "2020-01-04",
            "2020-01-04",
            "2020-01-01",
            "2020-01-01",
        ]
        index_data["acquisition_date"] = pd.to_datetime(index_data["acquisition_date"])
        index_data["prediction_time"] = pd.to_datetime(index_data["prediction_time"])

        result_series = features.CustomerLifetime().transform(index_data)
        index_series = pd.Series([1, 2], name="sample_id")
        expected_series = pd.Series(
            [7673.0, 0.0], name="customer_lifetime", index=index_series
        )

        pd.testing.assert_series_equal(result_series, expected_series)

    def test_customerlifetime_example_data(self, example_data):
        """
        GIVEN a CustomerLifetime-Feature instance and some example sales data
        WHEN the feature is applied to the data
        THEN the result is the lifetime of the customer at prediction time in days.
        """
        feature = features.CustomerLifetime()
        result_frame = feature.transform(example_data)

        expected_frame = (
            example_data.groupby(self.SAMPLE_LEVEL_INDEX_COLUMNS)
            .agg({"prediction_time": "first", "acquisition_date": "first"})
            .pipe(
                features.ODSFeature.calculate_time_difference,
                ref_date="acquisition_date",
                date_column="prediction_time",
                unit="1d",
            )
        ).rename("customer_lifetime")

        pd.testing.assert_series_equal(result_frame, expected_frame)
