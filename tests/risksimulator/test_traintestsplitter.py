"""Tests for the kundenscore.datasetconstructor.traintestsplitter module."""
import numpy as np
import pandas as pd
import pytest
import yaml

from kundenscore.datasetconstructor.traintestsplitter import GroupSplitter

SEEDS = [1, 6, 89, 12, 5]


class TestGroupSplitter:
    """Test suite for the GroupSplitter."""

    @pytest.fixture
    def group_splitter(self):
        return GroupSplitter(test_share=0.3, group_col="group_col", random_state=42)

    @pytest.fixture
    def dummy_data(self):
        data = pd.DataFrame(np.random.randn(1000, 10))
        data.columns = [f"col_{i}" for i in range(data.shape[1])]
        data["group_col"] = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], replace=True, size=1000
        )
        return data

    def test_instantiation(self):
        """
        WHEN a new GroupSplitter is instantiated with a test_share, group_col and seed
        THEN no exception is raised
        """
        GroupSplitter(test_share=0.3, group_col="group_col", random_state=42)

    def test_instantiation_wo_random_state_fails(self):
        """
        WHEN a new GroupSplitter is instantiated with a test_share and a group_col
        THEN no exception is raised
        """
        with pytest.raises(TypeError):
            GroupSplitter(test_share=0.3, group_col="group_col")

    @pytest.mark.parametrize(
        "config", [dict(test_share=0.3, group_col="col", random_state=52)],
    )
    def test_instantiation_from_config(self, config):
        """
        WHEN a new GroupSplitter is instantiated from a config
        THEN no exception is raised
        """
        GroupSplitter.from_config(config)

    @pytest.mark.parametrize(
        "params",
        [
            dict(test_share=0.3, random_state=42),
            dict(group_col="group_col", random_state=52),
        ],
    )
    def test_instantiation_with_missing_required_param_fails(self, params):
        """
        WHEN a new GroupSplitter is instantiated with a missing parameter
        THEN a TypeError is raised
        """
        with pytest.raises(TypeError):
            GroupSplitter(**params)

    def test_to_dict_output_is_yaml_serializable(self, group_splitter):
        """
        GIVEN a group splitter
        WHEN its to_dict method is called
        THEN the output can be serialized as YAML
        """
        yaml.dump(group_splitter.to_dict())

    @pytest.mark.parametrize(
        "params",
        [
            dict(test_share=0.3, group_col="group_col", random_state=42),
            dict(test_share=0.3, group_col="group_col", random_state=60),
        ],
    )
    def test_to_dict(self, params):
        """
        GIVEN a group splitter instantiated with keyword arguments
        WHEN the splitter's to_dict method is called
        THEN the returned dictionary representation holds the correct values for the \
            splitter's attributes
        AND the representation equals the keyword arguments used for instantiation
        """
        group_splitter = GroupSplitter(**params)

        representation = group_splitter.to_dict()

        assert representation["test_share"] == group_splitter.test_share
        assert representation["group_col"] == group_splitter.group_col
        if params.get("random_state"):
            assert representation["random_state"] == group_splitter.random_state
        else:
            with pytest.raises(KeyError):
                representation["random_state"]

        assert params == representation

    def test_instantiation_from_to_dict_output(self, group_splitter):
        """
        GIVEN a group splitter
        WHEN a new group splitter is instantiated from its dictionary representation
        THEN both splitters are equal
        AND both splitters' to_dict-output is equal
        """
        representation = group_splitter.to_dict()
        clone = GroupSplitter.from_config(representation)

        assert clone == group_splitter
        assert clone.to_dict() == group_splitter.to_dict()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reproduceability_within_splitter(self, seed, dummy_data):
        """
        GIVEN a GroupSplitter instantiated with a random_state and dummy data
        WHEN its split method is called twice with the data
        THEN both results are equal
        """
        splitter = GroupSplitter(
            test_share=0.3, group_col="group_col", random_state=seed
        )
        first_train, first_test = splitter.split(dummy_data)
        second_train, second_test = splitter.split(dummy_data)

        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_test, second_test)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reproduceability_between_splitters(self, seed, dummy_data):
        """
        GIVEN two GroupSplitters instantiated with the same params  and seed
        WHEN their split method is called with the data
        THEN both splitters' results are equal
        """
        test_share = 0.3
        group_col = "group_col"
        first_splitter = GroupSplitter(
            test_share=test_share, group_col=group_col, random_state=seed
        )
        second_splitter = GroupSplitter(
            test_share=test_share, group_col=group_col, random_state=seed
        )

        first_train, first_test = first_splitter.split(dummy_data)
        second_train, second_test = second_splitter.split(dummy_data)

        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_test, second_test)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_overlap_of_group_labels(self, group_splitter, dummy_data, seed):
        """
        GIVEN a group splitter and dummy data
        WHEN the splitter is applied to the data
        THEN the resulting training and test set do not overlap in terms of group labels
        """
        group_splitter.random_state = seed
        train, test = group_splitter.split(dummy_data)
        assert not np.any(train.group_col.isin(test.group_col))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_splitter_shuffles(self, group_splitter, dummy_data, seed):
        """
        GIVEN a group splitter and dummy data
        WHEN the splitter is applied to the data
        THEN the order of group labels in the result is shuffled
        """
        group_splitter.random_state = seed
        train, test = group_splitter.split(dummy_data)
        concatenated = pd.concat([train, test], axis=0)
        # unique does not sort
        assert np.any(concatenated.group_col.unique() != dummy_data.group_col.unique())
