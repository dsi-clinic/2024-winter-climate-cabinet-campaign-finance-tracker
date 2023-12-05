from abc import ABC, abstractmethod

import pandas as pd


class StateCleaner(ABC):
    """
    This abstract class is the one that all the state cleaners will be built on
    """

    @property
    def entity_name_dictionary(self) -> dict:
        """
        A dict mapping a state's raw entity names to standard versions

            Inputs: None

            Returns: _entity_name_dictionary
        """
        return self.entity_name_dictionary

    @abstractmethod
    def preprocess(self, filepaths_list: list[str]) -> list[pd.DataFrame]:
        """
        Preprocesses the state data and returns a dataframe

        Reads in the state's data, makes any necessary bug fixes, and
        combines the data into a list of DataFrames, discards data not schema

        Inputs:
            filepaths_list: list of absolute filepaths to relevant state data.
                required naming conventions, order, and extensions
                defined per state.

        Returns: a list of dataframes. If state data is all in one format
            (i.e. there are not separate individual and transaction tables),
            a list containing a single dataframe. Otherwise a list of three
            DataFrames that represent [transactions, individuals, organizations]
        """
        pass

    @abstractmethod
    def clean(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Cleans the state dataframe as needed and returns the dataframe

        Cleans the columns, converts dtypes to match database schema, and drops rows
        not representing minimal viable transactions

        Inputs:
            data: a list of 1 or 3 dataframes as outputted from preprocess method.

        Returns: a list of dataframes. If state data is all in one format
            (i.e. there are not separate individual and transaction tables),
            a list containing a single dataframe. Otherwise a list of three
            DataFrames that represent [transactions, individuals, organizations]
        """
        pass

    @abstractmethod
    def standardize(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Standardizes the dataframe into the necessary format for the schema

        Maps entity/office types and column names as defined in schema, adjust and add
        UUIDs as necessary

        Inputs:
            data: a list of 1 or 3 dataframes as outputted from clean method.

        Returns: a list of dataframes. If state data is all in one format
            (i.e. there are not separate individual and transaction tables),
            a list containing a single dataframe. Otherwise a list of three
            DataFrames that represent [transactions, individuals, organizations]
        """
        pass

    @abstractmethod
    def create_tables(
        self, data: list[pd.DataFrame]
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Creates the Individuals, Organizations, and Transactions tables from
        the dataframe list outputted from standardize

        Inputs:
            data: a list of 1 or 3 dataframes as outputted from standardize method.

        Returns: (individuals_table, organizations_table, transactions_table)
                    tuple containing the tables as defined in database schema
        """
        pass

    @abstractmethod
    def clean_state(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Runs the StateCleaner pipeline returning a tuple of cleaned dataframes

        Returns: use preprocess, clean, standardize, and create_tables methods
        to output (individuals_table, organizations_table, transactions_table)
        as defined in database schema
        """
        pass
