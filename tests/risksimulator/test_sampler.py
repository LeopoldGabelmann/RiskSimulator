"""Tests for the kundenscore.datasetconstructor.sampler module."""
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import yaml

from kundenscore.datasetconstructor.sampler import BinnedUniformSampler


class TestBinnedUniformSampler:
    """Test class for the binned uniform sampler"""

    @pytest.fixture(autouse=True)
    def seed_numpy(self):
        """Seed numpy with a random seed."""
        np.random.seed(42)

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """A fixture to provide a configuration to instantiate a BinnedUniformSampler."""
        return dict(
            min_date="2018-01-01",
            max_date="2019-12-31",
            lead_time="28d",
            prediction_period="180d",
            lookback="1y",
            samples_per_lookback=3,
        )

    def generate_data_for_one_customer(
        self,
        id: int,
        min_date: str,
        max_date: str,
        acquisition_date: str = None,
        n_orders: int = None,
        equispaced: bool = False,
    ) -> pd.DataFrame:
        """Generate dummy sales data for one customer.

        :param id: The id of the customer.
        :type id: int
        :param min_date: The lower boundary for the order date.
        :type min_date: Any other format accepted by `pd.to_datetime`.
        :param max_date: The upper boundary for the order date.
        :type max_date: Any format accepted by `pd.to_datetime`
        :param acquisition_date: The acuquisition date for the customer. Will be chosen \
            randomly if not provided. Will overwrite min_date if min_date < acquisition_date.
        :type acquisition_date: Any format accepted by `pd.to_datetime`
        :param n_orders: Number of orders to generate for the customer. Will be chosen \
            randomly if not provided.
        :type n_orders: int
        :param equispaced: Shall the orders be equispaced or sampled from a uniform \
            distribution over the interval `[min_date, max_date]`?
        :type equispaced: bool
        :return: A dummy purchase history holding the columns \
            `[order_date, customer_id, acquisition_date, orderposition_id]`
        :rtype: pd.DataFrame
        """

        if not acquisition_date:
            min_ac_date = pd.to_datetime("1980-01-01")
            max_ac_date = pd.to_datetime(max_date)
            acquisition_date = min_ac_date + np.random.uniform() * (
                max_ac_date - min_ac_date
            )
        acquisition_date = pd.to_datetime(acquisition_date)

        min_date = max(acquisition_date, pd.to_datetime(min_date))
        max_date = pd.to_datetime(max_date)

        if not max_date > min_date:
            raise ValueError("max_date must be greater min_date.")

        if not n_orders:
            # one order per month
            n_orders = (max_date - min_date) / pd.Timedelta("30d")
            # reduce by random factor but keep at least one order
            n_orders = np.ceil(n_orders * np.random.uniform()).astype(int)

        delta = max_date - min_date

        if equispaced:
            order_dates = min_date + pd.to_timedelta(
                np.linspace(0, 1, n_orders) * delta
            )
        else:
            order_dates = min_date + pd.to_timedelta(
                np.random.uniform(size=n_orders) * delta
            )

        data = pd.DataFrame()
        data["order_date"] = order_dates
        data["customer_id"] = id
        data["acquisition_date"] = acquisition_date
        data["orderposition_id"] = np.arange(data.shape[0]) + 1
        return data

    @pytest.fixture
    def sampler(self, config: Dict[str, Any]) -> BinnedUniformSampler:
        """Fixture that provides an instantiated sampler.

        :param config: The config-fixture for the sampler.
        :type config: Dict[str,Any]
        :return: The instantiated sampler
        :rtype: BinnedUniformSampler
        """
        return BinnedUniformSampler.from_config(config)

    @pytest.fixture
    def raw_data(self) -> pd.DataFrame:
        """Generate dummy purchase data for 100 customers.

        :return: A dummy purchase history holding the columns \
            `[order_date, customer_id, acquisition_date, orderposition_id]` plus some columns \
                holding random feature data.
        :rtype: pd.DataFrame
        """

        min_date = "2016-01-01"
        max_date = "2019-12-13"
        raw_data = [
            self.generate_data_for_one_customer(i, min_date, max_date)
            for i in range(100)
        ]
        raw_data = pd.concat(raw_data, axis=0)
        for i in range(10):
            raw_data[f"feat_{i}"] = np.random.randn(raw_data.shape[0])
        return raw_data

    def test_instantiation_with_all_params(self, config):
        BinnedUniformSampler(**config)

    @pytest.mark.parametrize(
        "missing_field",
        [
            "min_date",
            "max_date",
            "lead_time",
            "prediction_period",
            "lookback",
            "samples_per_lookback",
        ],
    )
    def test_instantiation_with_missing_field_fails(
        self, missing_field: str, config: Dict[str, Any]
    ):
        """
        GIVEN a config with a missing specification for a parameter. 
        WHEN a BinnedUniformSampler is instantiated with the missing keyword argument.
        THEN a TypeError is raised
        """
        del config[missing_field]
        with pytest.raises(expected_exception=TypeError):
            _ = BinnedUniformSampler(**config)

    @pytest.mark.parametrize(
        "param,value_wrong_type",
        [
            ("max_date", "1y"),
            ("prediction_period", "2019-06-01"),
            ("lead_time", "2018-01-01"),
            ("lookback", "2018-01-01"),
            ("samples_per_lookback", "100"),
        ],
    )
    def test_instantiation_with_wrong_data_type_fails(
        self, param: str, value_wrong_type: Any, config: Dict[str, Any]
    ):
        """
        GIVEN a keyword argument's value of the wrong data type.
        WHEN a BinnedRandomSampler is instantiated with the wrong value
        THEN a ValueError is raised.
        """

        config[param] = value_wrong_type
        with pytest.raises(ValueError):
            BinnedUniformSampler(**config)

    def test_instantiation_from_config(self, config: Dict[str, Any]):
        """
        GIVEN a valid config for a BinnedUniformSampler
        WHEN the classmethod `from_config` is called with this config
        THEN a BinnedUniformSampler is instantiated.
        """
        BinnedUniformSampler.from_config(config)

    def test_to_dict_is_yaml_serializable(self, sampler: BinnedUniformSampler):
        """
        GIVEN an instantiated BinnedUniformSampler
        WHEN the instance's `to_dict` method is called and the output is dumped to YAML
        THEN no exception is raised.
        """
        yaml.dump(sampler.to_dict())

    def test_equals_w_equal_instances(self, config: Dict[str, Any]):
        """
        GIVEN keyword arguments for a BinnedUniformSampler
        WHEN two samplers are instantiated from the same config
        THEN they are equal.
        """
        first_sampler = BinnedUniformSampler(**config)
        second_sampler = BinnedUniformSampler(**config)
        assert first_sampler == second_sampler

    def test_equals_wo_equal_instance(
        self, sampler: BinnedUniformSampler, config: Dict[str, Any]
    ):
        """
        GIVEN a valid set of keyword arguments for a BinnedUniformSampler
        WHEN one sampler is instantiated with that config and another sampler is instantiated \
            with a slightly changed set of keyword arguments
        THEN the two samplers are not equal.
        """
        first_sampler = BinnedUniformSampler(**config)
        config["lookback"] = str(sampler.lookback / 2)
        second_sampler = BinnedUniformSampler(**config)
        assert first_sampler != second_sampler

    def test_instantiation_from_to_dict_output(self, sampler: BinnedUniformSampler):
        """
        GIVEN the output from a samplers `to_dict` method
        WHEN a new sampler is instantiated from this output via the `from_config` classmethod
        THEN the two resulting samplers are equal.
        """
        representation = sampler.to_dict()
        clone = BinnedUniformSampler.from_config(representation)
        assert clone == sampler

    @pytest.mark.parametrize(
        "expected_column",
        ["prediction_time", "x_include", "y_include", "sample_wo_data", "sample_id"],
    )
    def test_sampled_data_columns(
        self,
        sampler: BinnedUniformSampler,
        expected_column: str,
        raw_data: pd.DataFrame,
    ):
        """
        GIVEN a sampler and dummy purchase data for multiple customers
        WHEN samples are created from this dummy data using the sampler
        THEN the output dataframe from the sampler contains all mandatory columns.
        """
        sampled = sampler.generate_samples(raw_data)
        columns = list(sampled.columns) + list(sampled.index.names)
        assert expected_column in columns

    def test_no_data_in_lead_time(
        self, sampler: BinnedUniformSampler, raw_data: pd.DataFrame
    ):
        """
        GIVEN a sampler with a lead time and some dummy purchase data to create samples from
        WHEN samples are generated from this purchase data
        THEN there is no order position marked in the column `y_include` in the lead time
        """
        sampled = sampler.generate_samples(raw_data)
        y_data = sampled[sampled.y_include]
        lead_time = sampler.lead_time
        lower_boundary = lead_time + y_data.prediction_time
        assert np.all(y_data.order_date > lower_boundary)

    def test_no_prediction_time_outside_min_and_max_date(
        self, sampler: BinnedUniformSampler, raw_data: pd.DataFrame
    ):
        """
        GIVEN a sampler instance and some dummy purchase data for multiple customers
        WHEN samples are created from this data using the sampler
        THEN the prediction times of all samples is within `[sampler.min_date, sampler.max_date]`
        """
        sampled = sampler.generate_samples(raw_data)
        max_date = sampler.max_date
        min_date = sampler.min_date
        assert np.all(sampled.prediction_time > min_date)
        assert np.all(sampled.prediction_time < max_date)

    def test_all_x_data_in_lookback(self, sampler, raw_data):
        """
        GIVEN a sampler instance and some dummy purchase data for multiple customers
        WHEN samples are generated from the data
        THEN all the order positions relevant for the features are within the lookback \ 
            period as calculated from their prediction times.
        """
        sampled = sampler.generate_samples(raw_data)
        x_data = sampled[sampled.x_include]
        lookback = sampler.lookback

        lower_boundary = x_data.prediction_time - lookback
        upper_boundary = x_data.prediction_time

        assert np.all(x_data.order_date > lower_boundary)
        assert np.all(x_data.order_date < upper_boundary)

    def test_all_y_data_in_prediction_period(self, sampler, raw_data):
        """
        GIVEN a sampler instance and some dummy purchase data for multiple customers
        WHEN samples are generated from the data
        THEN all the order positions relevant for the target variable are within the period \ 
            defined by the lead time and the prediction time of the sample.
        """
        sampled = sampler.generate_samples(raw_data)
        y_data = sampled[sampled.y_include]
        lead_time = sampler.lead_time
        prediction_period = sampler.prediction_period

        lower_boundary = y_data.prediction_time + lead_time
        upper_boundary = lower_boundary + prediction_period

        assert np.all(y_data.order_date > lower_boundary)
        assert np.all(y_data.order_date < upper_boundary)

    def test_one_prediction_time_per_sample(self, sampler, raw_data):
        """
        GIVEN a sampler instance and some dummy purchase data for multiple customers
        WHEN samples are generated from the data
        THEN each sample has exactly one unique value for its prediction time.
        """
        sampled = sampler.generate_samples(raw_data)
        n_prediction_times = sampled.groupby("sample_id").prediction_time.nunique()
        assert np.all(n_prediction_times == 1)

    def test_one_customer_per_sample(self, sampler, raw_data):
        """
        GIVEN a sampler instance and some dummy purchase data for multiple customers
        WHEN samples are generated from the data
        THEN each sample is based on exactly one customer's data.
        """
        sampled = sampler.generate_samples(raw_data)
        n_customers = sampled.groupby("sample_id").customer_id.nunique()
        assert np.all(n_customers == 1)

    @pytest.mark.parametrize("min_date", ["2016-01-01"])
    @pytest.mark.parametrize("max_date", ["2019-12-31"])
    @pytest.mark.parametrize("lead_time", ["14d", "28d"])
    @pytest.mark.parametrize("lookback", ["2y", "180d"])
    @pytest.mark.parametrize("prediction_period", ["30d", "180d"])
    @pytest.mark.parametrize("samples_per_lookback", [3, 12])
    @pytest.mark.parametrize("random_seed", [42, 3, 19])
    def test_samples_within_bins(
        self,
        min_date,
        max_date,
        lead_time,
        prediction_period,
        lookback,
        samples_per_lookback,
        random_seed,
    ):
        """
        GIVEN a set of parameters for a sampler
        WHEN samples are generated from one customers data using a sampler instantiated \
            with these parameters
        THEN all samples for this customers fall into equi-spaced bins of the customers's \
            purchase history.
        """
        np.random.seed(random_seed)

        # generate data for timespan
        customer_data = self.generate_data_for_one_customer(1, min_date, max_date)

        # initialize sampler
        sampler = BinnedUniformSampler(
            min_date=min_date,
            max_date=max_date,
            lead_time=lead_time,
            prediction_period=prediction_period,
            lookback=lookback,
            samples_per_lookback=samples_per_lookback,
        )
        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)
        lead_time = pd.to_timedelta(lead_time)
        prediction_period = pd.to_timedelta(prediction_period)
        lookback = pd.to_timedelta(lookback)

        samples = sampler.generate_samples(customer_data)

        # calculate boundaries of the sampling range
        upper = max_date - prediction_period - lead_time
        lower = max(min_date, customer_data.acquisition_date.max())

        # calculate the number of samples
        lookbacks_covered = (upper - lower) / lookback
        n_samples_expected = np.floor(lookbacks_covered * samples_per_lookback)
        n_samples_expected = n_samples_expected.astype(int)
        # at least one sample if customer has enough data for one prediction period
        if upper > lower:
            n_samples_expected = max(n_samples_expected, 1)
        else:
            n_samples_expected = 0

        # full lookbacks for customers with enough data
        if lower < (upper - lookback):
            lower = lower + lookback
        lower = max(lower, min_date + lookback)

        # calculate the size of the individual bins for sampling
        bin_size = (upper - lower) / n_samples_expected

        # sort prediction times into bins
        prediction_times = samples.groupby("sample_id").prediction_time.first()
        bins = np.floor((prediction_times - lower) / bin_size)

        # check that we actually generated the expected number of samples
        assert (
            samples.index.get_level_values("sample_id").nunique() == n_samples_expected
        )
        # check that every sample falls into its own bin
        assert bins.nunique() == n_samples_expected

    @pytest.mark.parametrize("min_date", ["2016-01-01"])
    @pytest.mark.parametrize("max_date", ["2019-12-31"])
    @pytest.mark.parametrize("lookback", ["2y", "180d"])
    @pytest.mark.parametrize("prediction_period", ["30d", "180d"])
    @pytest.mark.parametrize("lead_time", ["14d", "28d"])
    @pytest.mark.parametrize("samples_per_lookback", [3, 12])
    def test_samples_for_new_customers(
        self,
        min_date,
        max_date,
        lookback,
        prediction_period,
        lead_time,
        samples_per_lookback,
    ):
        """
        GIVEN a set of parameters for a sampler
        WHEN samples are generated from a customer's purchase history that does not cover a \
            full lookback
        THEN the number of samples generated for this customer is still proportional to the \
            time period covered by the customers purchase history and there is at least one \
            sample regardless of the available purchase history.
        """
        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)
        lookback = pd.to_timedelta(lookback)
        prediction_period = pd.to_timedelta(prediction_period)
        lead_time = pd.to_timedelta(lead_time)
        samples_per_lookback = 3

        acquisition_date = (
            max_date - (lookback * np.random.uniform()) - prediction_period - lead_time
        )
        new_customer_data = self.generate_data_for_one_customer(
            1, min_date, max_date, acquisition_date
        )

        sampler = BinnedUniformSampler(
            min_date=min_date,
            max_date=max_date,
            lead_time=lead_time,
            prediction_period=prediction_period,
            lookback=lookback,
            samples_per_lookback=samples_per_lookback,
        )
        samples = sampler.generate_samples(new_customer_data)

        n_samples = samples.index.get_level_values("sample_id").nunique()

        lookbacks_covered = (
            (max_date - prediction_period - lead_time) - acquisition_date
        ) / lookback
        expected_n_samples = np.max(
            [np.floor(lookbacks_covered * samples_per_lookback), 1]
        )
        assert expected_n_samples == n_samples
        assert np.all(n_samples > 0)

    def test_x_data_but_no_y_data(self):
        """
        GIVEN a sampler and dummy data for a customer with no purchases falling into the \
            prediction period
        WHEN samples are generated from that customers data
        THEN there are no orderpositions marked with `y_include` in the result. 
        """
        lead_time = pd.to_timedelta("1d")
        lookback = pd.to_timedelta("2y")
        prediction_period = pd.to_timedelta("180d")
        min_date = pd.to_datetime("2018-01-01")
        max_date = (
            min_date + lookback + lead_time + prediction_period + pd.Timedelta("1d")
        )

        sampler = BinnedUniformSampler(
            min_date=min_date,
            max_date=max_date,
            lead_time=lead_time,
            prediction_period=prediction_period,
            samples_per_lookback=1,
            lookback=lookback,
        )
        # purchase that covers one lookback, but no prediction period
        customer_data = self.generate_data_for_one_customer(
            1, min_date, min_date + lookback, n_orders=12
        )
        # a sampler with max date greater than the maximum order date and
        # params such that exactly one sample is created for the customer
        samples = sampler.generate_samples(customer_data)

        assert samples.index.get_level_values("sample_id").nunique() == 1
        assert samples.x_include.sum() >= 1
        assert samples.y_include.sum() == 0

    def test_y_data_but_no_x_data(self):
        """
        GIVEN a sampler and dummy data for a customer with no purchases falling into the \
            lookback period
        WHEN samples are generated from that customers data
        THEN there are no orderpositions marked with `x_include` in the result. 
        """
        lead_time = pd.to_timedelta("1d")
        lookback = pd.to_timedelta("2y")
        prediction_period = pd.to_timedelta("180d")
        max_date = pd.to_datetime("2020-01-01")
        min_date = (
            max_date - prediction_period - lead_time - lookback - pd.to_timedelta("1d")
        )

        sampler = BinnedUniformSampler(
            min_date=min_date,
            max_date=max_date,
            lead_time=lead_time,
            prediction_period=prediction_period,
            samples_per_lookback=1,
            lookback=lookback,
        )

        # purchase that covers one prediction period, but no lookback
        customer_data = self.generate_data_for_one_customer(
            1, max_date - prediction_period - lead_time, max_date, n_orders=12
        )

        # a sampler with max date greater than the maximum order date and
        # params such that exactly one sample is created for the customer
        samples = sampler.generate_samples(customer_data)

        assert samples.index.get_level_values("sample_id").nunique() == 1
        assert samples.x_include.sum() == 0
        assert samples.y_include.sum() >= 1

    def test_acquisition_date_greater_max_date(self):
        """
        GIVEN a sampler and data for a customer whose acquisition date is greater \
            than the samplers `max_date`
        WHEN the sampler is applied to the data
        THEN it returns an empty dataframe
        """
        customer_data = self.generate_data_for_one_customer(
            1,
            min_date="2016-01-01",  # will be overwritten by acquisition date
            max_date="2020-08-01",
            acquisition_date="2020-01-01",
            n_orders=12,
        )
        sampler = BinnedUniformSampler(
            min_date="2016-01-01",
            max_date="2019-12-31",
            lead_time="28d",
            prediction_period="180d",
            lookback="180d",
            samples_per_lookback=1,
        )
        samples = sampler.generate_samples(customer_data)

        assert samples.empty

    @pytest.mark.parametrize("seed", [1, 218, 39])
    def test_result_is_reproduceable_between_samplers(self, config, raw_data, seed):
        """
        GIVEN two samplers with the same random seed and the same other parameters 
        WHEN both samlers are applied to the same raw data
        THEN the result is the same
        """
        first_sampler = BinnedUniformSampler(random_seed=seed, **config)
        first_result = first_sampler.generate_samples(raw_data)

        second_sampler = BinnedUniformSampler(random_seed=seed, **config)
        second_result = second_sampler.generate_samples(raw_data)

        pd.testing.assert_frame_equal(first_result, second_result)

    @pytest.mark.parametrize("seed", [1, 218, 39])
    def test_result_is_reproduceable_within_sampler(self, config, raw_data, seed):
        """
        GIVEN a sampler instantiated with a random seed
        WHEN this sampler is applied to the same data twice
        THEN the results of both applications are equal
        """
        sampler = BinnedUniformSampler(random_seed=seed, **config)

        first_result = sampler.generate_samples(raw_data)
        second_result = sampler.generate_samples(raw_data)

        pd.testing.assert_frame_equal(first_result, second_result)

    @pytest.mark.parametrize("seed", [1, 218, 39])
    def test_different_seeds_produce_different_outcomes(self, config, raw_data, seed):
        """
        GIVEN two samplers with the same parameters but different random seeds
        WHEN both samplers are applied to the same raw data
        THEN the result is different
        """
        first_sampler = BinnedUniformSampler(random_seed=seed, **config)
        second_sampler = BinnedUniformSampler(random_seed=seed + 1, **config)

        first_result = first_sampler.generate_samples(raw_data)
        second_result = second_sampler.generate_samples(raw_data)

        try:
            pd.testing.assert_frame_equal(first_result, second_result)
        except AssertionError:
            pass
        else:
            raise AssertionError
