import numpy as np
import pandas as pd
import pytest
import mock
import yaml

from kundenscore.datasetconstructor.featurecollection import FeatureCollection


class TestFeatureCollection:
    """A test suite for kundenscore.datasetconstructor.featurecollection.FeatureCollection."""

    @pytest.fixture
    def mocked_features(self):
        """Get a list of mocked feature objects."""
        mocked_features = list()
        for i in range(10):
            name = f"feature_{i}"
            mock_feature = mock.Mock(name=name)
            mock_feature.transform.return_value = pd.Series(
                np.random.randn(100), name=name
            )
            mocked_features.append(mock_feature)
        return mocked_features

    @pytest.fixture
    def dummy_data(self):
        """Get a dummy dataframe for testing."""
        return pd.DataFrame(np.random.randn(100, 10))

    def test_instantiation_with_features(self, mocked_features):
        """
        GIVEN a list of mocked features
        WHEN a feature collection is instantiated with these features
        THEN its features are the mocked features
        """
        collection = FeatureCollection(mocked_features)
        assert collection.features == mocked_features

    def test_instantiation_without_features_fails(self):
        """
        GIVEN an empty list
        WHEN a feature collection is instantiated with this emtpy list
        THEN a ValueError is raised.
        """
        features = list()
        with pytest.raises(ValueError):
            FeatureCollection(features)

    def test_generate_features_calls_all_transform_methods_once(
        self, mocked_features, dummy_data
    ):
        """
        GIVEN a feature collection instantiated with some mocked features
        WHEN the collection's `generate_features` method is called
        THEN the transform method of each mocked feature gets called once
        AND the result returned by the collection equals the concatenated results \
            of all mocked features
        """
        collection = FeatureCollection(mocked_features)

        result = collection.generate_features(dummy_data)

        for f in mocked_features:
            f.transform.assert_called_once()
            input_df = f.transform.call_args[0][0]
            pd.testing.assert_frame_equal(dummy_data, input_df)

        expected = pd.concat([f.transform() for f in mocked_features], axis=1)

        pd.testing.assert_frame_equal(result, expected)

    @mock.patch("kundenscore.datasetconstructor.features.Feature", spec=True)
    def test_to_dict_returns_class_names(self, mocked_feature):
        """
        GIVEN a feature collection that is instantiated with one mocked feature
        WHEN the collections `to_dict`-method is called
        THEN the result is a dictionary holding the features class name
        """
        collection = FeatureCollection([mocked_feature])
        representation = collection.to_dict()
        assert representation == {"features": ["Feature"]}

    @mock.patch("kundenscore.datasetconstructor.features.Feature", spec=True)
    def test_to_dict_is_yaml_serializable(self, mock_feature):
        """
        GIVEN a feature collection that is instantiated with one mocked feature
        WHEN the collection's `to_dict` method is called
        THEN the result is serializable to yaml
        """
        collection = FeatureCollection([mock_feature])
        yaml.dump(collection.to_dict())

    def test_from_config_instantiates_features(self):
        """
        GIVEN a config that holds some features class names
        WHEN a feature collection is instantiated from this config 
        THEN the constructor of all features is called once
        AND the feature collection holds the features from the config.
        """

        # mock some features in the kundenscore.datasetconstructor.features module
        feature_class_names = ["FirstFeature", "SecondFeature", "ThirdFeature"]
        mock_module = mock.Mock()
        mocked_features = list()
        for f_name in feature_class_names:
            mock_feature = mock.Mock(name=f_name)
            getattr(mock_module, f_name).return_value = mock_feature
            mocked_features.append(mock_feature)

        with mock.patch(
            "kundenscore.datasetconstructor.featurecollection.features", new=mock_module
        ):
            config = {"features": feature_class_names}
            collection = FeatureCollection.from_config(config)

            assert collection.features == mocked_features

            # assert all constructors have been called once
            for f_name in feature_class_names:
                getattr(mock_module, f_name).assert_called_once()
