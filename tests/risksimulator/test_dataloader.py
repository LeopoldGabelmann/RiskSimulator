"""Unit tests for the data loader."""
import os
import uuid

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as db
import yaml

from kundenscore import settings
from kundenscore.datasetconstructor.dataloader import H5DataLoader
from kundenscore.datasetconstructor.dataloader import SQLDataLoader


class TestSQLDataLoader:
    """Contains all tests regarding the SQLDataLoader."""

    @pytest.fixture
    def sqldataloader_config(self,):
        """Provide a config file for the SQLDataLoader."""
        init_config = dict()
        init_config["path_sql_query"] = "test_path"
        init_config["dialect"] = "test_dialect"
        init_config["host"] = "test_host"
        return init_config

    @pytest.fixture
    def sqldataloader_init(self, sqldataloader_config):
        """Instantiate the SQLDataLoader."""
        return SQLDataLoader(**sqldataloader_config)

    @pytest.fixture
    def mock_env_missing(self, monkeypatch):
        """
        Set the environment variable containing the connection information is missing.
        """
        monkeypatch.delenv("SQLDATALOADER_USERNAME", raising=False)
        monkeypatch.delenv("SQLDATALOADER_PASSWORD", raising=False)

    @pytest.fixture
    def mock_env_user(self, monkeypatch):
        """Set environment variables.

        Set the environment variables defining the user and the password
            to for the sql connection.
        """
        monkeypatch.setenv("SQLDATALOADER_USERNAME", "TestUser")
        monkeypatch.setenv("SQLDATALOADER_PASSWORD", "TestPassword")

    @pytest.fixture
    def config_test_database(self,):
        """
        Create a dictionary yielding all connection parameters to the sqlite
            test database.
        """
        init_config = dict()
        init_config["path_sql_query"] = (
            settings.TEST_PATH / "data" / "test_sql_query.sql"
        )
        init_config["dialect"] = "sqlite"
        init_config["host"] = "testhost"
        return init_config

    @pytest.fixture
    def dummy_data(self,):
        """ Generate dummy data for the test database to be written and loaden. """
        return [
            {"Id": "2", "name": "ram", "salary": 80000, "active": 0},
            {"Id": "3", "name": "ramesh", "salary": 70000, "active": 1},
        ]

    @pytest.fixture
    def dummy_dataframe(self, dummy_data):
        """Convert dummy data into a dummy data frame."""
        dataframe = pd.DataFrame(dummy_data).astype(
            {"Id": "int64", "name": "str", "salary": "float64", "active": "int64"}
        )
        return dataframe

    @pytest.fixture
    def sqlite_database(self, dummy_data):
        """Build a small test sqlite database with two entries and a unique name."""
        # Get unique_name and set it as a suffix.
        unique_name = str(uuid.uuid4())
        database_path = (
            settings.TEST_PATH / "data" / f"temp_sqlite_{unique_name}.sqlite"
        )
        eng = db.create_engine(f"sqlite:///{database_path}")
        metadata = db.MetaData()
        connection = eng.connect()
        emp = db.Table(
            "emp",
            metadata,
            db.Column("Id", db.Integer()),
            db.Column("name", db.String(255), nullable=False),
            db.Column("salary", db.Float(), default=100.0),
            db.Column("active", db.Integer()),
        )
        metadata.create_all(eng)

        entry_query = db.insert(emp)
        values_list = dummy_data
        connection.execute(entry_query, values_list)
        connection.close()

        yield database_path

        os.remove(database_path)

    def test_loader_instantiation(self, sqldataloader_config):
        """
        GIVEN a correct config file.
        WHEN the SQLDataLoader is instantiated with that config file
        THEN the SQLDataLoader is instantiated.
        """
        SQLDataLoader(**sqldataloader_config)

    def test_config_output_equal_input(self, sqldataloader_config):
        """
        GIVEN a correct config file.
        WHEN the config file is recreated by the method 'to_dict'
        THEN the two instantiated samplers are equal.
        """
        loader = SQLDataLoader(**sqldataloader_config)
        config_dict = loader.to_dict()
        assert loader == SQLDataLoader.from_config(config_dict)

    def test_config_output_serializable(self, sqldataloader_init):
        """
        GIVEN an instantiated SQLDataLoader.
        WHEN a config file is created by the method 'to_dict'
        THEN it can be saved as a yaml file.
        """
        yaml.dump(sqldataloader_init.to_dict())

    def test_pathlib_path_to_yaml(self, sqldataloader_config):
        """
        GIVEN a configuration dictonary.
        WHEN the path to the sql query is set to a pathlib path and a SQLDataLoader is
             instantiated with that configuration
             THEN it can be saves as a yaml file.
        """
        sqldataloader_config["path_sql_query"] = (
            settings.TEST_PATH / "data" / "test_sql_query.sql"
        )
        loader = SQLDataLoader.from_config(sqldataloader_config)
        yaml.dump(loader.to_dict())

    @pytest.mark.parametrize(
        "init_param, new_value",
        [
            ("path_sql_query", 100.35),
            ("host", [10.0, False]),
            ("dialect", [13.3, False]),
        ],
    )
    def test_wrong_init_type(self, sqldataloader_config, init_param, new_value):
        """
        GIVEN wrong input types of the dictionary params for the SQLDataLoader.
        WHEN the SQLDataLoader is instantiated
        THEN a TypeError is raised.
        """
        sqldataloader_config[init_param] = new_value
        with pytest.raises(TypeError):
            SQLDataLoader(**sqldataloader_config)

    def test_load_testquery(self, sqldataloader_config):
        """
        GIVEN a sql test file and a correct path.
        WHEN the SQLDataLoader is instantiated with that path and the method
            'read_query' is called
        THEN the query is read in correctly.
        """
        sqldataloader_config["path_sql_query"] = (
            settings.TEST_PATH / "data" / "test_sql_query.sql"
        )
        loader = SQLDataLoader(**sqldataloader_config)
        query = loader._read_query()
        assert query == "SELECT Id, name, salary, active FROM emp"

    def test_nonexistent_query_path(self, sqldataloader_config):
        """
        GIVEN an instantiated SQLDataLoader with the wrong path to the sql file.
        WHEN the method 'read_query' is called
        THEN a NameError is raised.
        """
        sqldataloader_config["path_sql_query"] = "wrong_test_path"
        with pytest.raises(NameError):
            SQLDataLoader.from_config(sqldataloader_config)._read_query()

    def test_no_environment_connection(self, sqldataloader_config, mock_env_missing):
        """
        GIVEN that the environment variable defining the connection to the
            sql database is missing and an instantiated SQLDataLoader
        WHEN the method 'connect_to_server' is called
        THEN an EnvironmentError is raised."""
        sqldataloader_config["user"] = True
        with pytest.raises(EnvironmentError):
            SQLDataLoader.from_config(sqldataloader_config)._connect_to_server()

    def test_connect_testdatabase(self, config_test_database, sqlite_database):
        """
        GIVEN an instantiated SQLDataLoader and an existing test sqlite database.
        WHEN the method 'connect_to_server' is evoked
        THEN the connection can be established.
        """
        config_test_database["host"] = str(sqlite_database)
        SQLDataLoader.from_config(config_test_database)._connect_to_server()

    def test_connect_set_environment_but_user_false(
        self, config_test_database, mock_env_user, sqlite_database
    ):
        """
        GIVEN set environment variables for the user and the password to
            connect to the sql database.
        WHEN a SQLDataLoader is instantiated with the user set to False
        THEN the connection to the sql database is established by
            ignoring the environment variables.
        """
        config_test_database["host"] = str(sqlite_database)
        config_test_database["user"] = False
        SQLDataLoader.from_config(config_test_database)._connect_to_server()

    def test_no_existence_sqlite(self, config_test_database):
        """
        GIVEN a dictionary configuring the SQLDataLoader with a sqlite
            dialect and an non existent host file.
        WHEN a SQLDataLoader is instantiated from that dict and tries to load the data
        THEN a ValueError is raised.
        """
        config_test_database["dialect"] = "sqlite"
        config_test_database["host"] = "test_host"

        with pytest.raises(ValueError):
            SQLDataLoader.from_config(config_test_database)._connect_to_server()

    def test_load_data(self, config_test_database, sqlite_database, dummy_dataframe):
        """
        GIVEN a dictionary defining the connection to the test sql database.
        WHEN a SQLDataLoader is instantiated and the method 'load_data' is used
        THEN it returns a dataframe equal to the dataframe of the sql test database.
        """
        config_test_database["host"] = str(sqlite_database)
        loader = SQLDataLoader.from_config(config_test_database)
        loaded_dataframe = loader.load_data()

        pd.testing.assert_frame_equal(loaded_dataframe, dummy_dataframe)

    def test_write_equals_load_data(
        self, config_test_database, sqlite_database, dummy_dataframe
    ):
        """
        GIVEN a dictionary defining the connection to the test sql database.
        WHEN a SQLDataLoader is instantiated and the method 'write_query_to_hdf' is
            used
        THEN it writes a dataframe to the test data folder and the resulting dataframe
            is the same then the read in dataframe.
        """
        test_data_path = settings.TEST_PATH / "data" / "test_raw_data.hdf"
        config_test_database["host"] = str(sqlite_database)
        loader = SQLDataLoader.from_config(config_test_database)
        loader.write_query_to_hdf(output_name=test_data_path)

        reloaded_data = pd.read_hdf(test_data_path, key="raw_data", mode="r+")
        os.remove(test_data_path)

        pd.testing.assert_frame_equal(reloaded_data, dummy_dataframe)


class TestH5DataLoader:
    """Tests for kundenscore.datasetconstructor.dataloader.H5DataLoader."""

    @pytest.fixture
    def example_data(self):
        """Some example data to load from hdf."""
        return pd.DataFrame({"col1": np.random.randn(2), "col2": ["hello", "world"]})

    @pytest.fixture
    def dataloader_config(self, example_data):
        """A config for a H5DataLoader that points to an existing H5 file."""
        if not os.path.exists(settings.TEST_TMP_DIR):
            os.makedirs(settings.TEST_TMP_DIR)
        filename = str(uuid.uuid4()) + ".h5"

        f_hdf = settings.TEST_TMP_DIR / filename
        f_hdf = f_hdf.absolute()
        key = "data"
        example_data.to_hdf(f_hdf, key)

        yield {"f_hdf": str(f_hdf), "key": key}

        os.remove(f_hdf)

    @pytest.fixture
    def dataloader(self, dataloader_config):
        return H5DataLoader(**dataloader_config)

    def test_from_config(self, dataloader_config):
        """
        GIVEN a valid config for a H5DataLoader
        WHEN it is instantiated from it
        THEN everything works fine.
        """
        H5DataLoader.from_config(dataloader_config)

    def test_load_data_returns_data(self, dataloader, example_data):
        """
        GIVEN an instantiated H5DataLoader configured with a valid path to a H5 file
        WHEN its load_data method is invoked
        THEN it returns the data from the file.
        """
        data = dataloader.load_data()
        pd.testing.assert_frame_equal(data, example_data)

    def test_to_dict(self, dataloader_config):
        """
        GIVEN a config for a H5DataLoader
        WHEN it is instantiated from it AND its to_dict method is invoked
        THEN initial config and the newly generated representation are equal.
        """
        loader = H5DataLoader.from_config(dataloader_config)
        representation = loader.to_dict()
        assert dataloader_config == representation

    def test_yaml_serialization_and_deserialization(self, dataloader_config):
        """
        GIVEN a config for a H5DataLoader
        WHEN the loader is instantiated from it AND the result of its to_dict method is
            serialized and de-serialized
        THEN the deserialized config equals the initial config.
        """
        loader = H5DataLoader.from_config(dataloader_config)

        representation = loader.to_dict()
        serialized = yaml.dump(representation)
        deserialized = yaml.load(serialized)

        assert deserialized == dataloader_config

    def test_to_and_from_yaml_is_the_same_data(self, dataloader_config, example_data):
        """
        GIVEN an instantiated H5DataLoader
        WHEN its to_dict method is invoked AND the result is serialized to and
            deserialized from YAML AND a new H5DataLoader is instantiated from the
            resulting config
        THEN both loaders have the same attributes AND both loaders load_data methods
            return the same data.
        """
        loader = H5DataLoader.from_config(dataloader_config)

        cloned_config = loader.to_dict()
        yaml_config = yaml.dump(cloned_config)
        deserialized_config = yaml.load(yaml_config)

        cloned_loader = H5DataLoader.from_config(deserialized_config)

        assert loader.f_hdf == cloned_loader.f_hdf
        assert loader.key == cloned_loader.key
        pd.testing.assert_frame_equal(loader.load_data(), cloned_loader.load_data())
