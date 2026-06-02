from abc import ABC, abstractmethod


class BaseFeature(ABC):
    """
    Abstract base class for all feature engineering modules.

    Every feature engineering class should inherit from this class
    and implement the required methods.

    Example:
    --------
    class GeohashFeatures(BaseFeature):

        def fit(self, df):
            ...

        def transform(self, df):
            ...

    Why?
    ----
    This ensures that all feature modules have a consistent interface,
    allowing them to be easily plugged into the FeaturePipeline.
    """

    @abstractmethod
    def fit(self, df):
        """
        Learn any statistics required for feature generation.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset.

        Returns
        -------
        self
            Returns the fitted feature object.

        Examples
        --------
        - Learn geohash frequency counts.
        - Learn target encoding mappings.
        - Fit label encoders.
        - Compute global means/medians.

        Notes
        -----
        This method should NEVER modify the input dataframe.
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Generate features using previously learned statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset (train, validation, or test).

        Returns
        -------
        pd.DataFrame
            DataFrame containing newly generated features.

        Examples
        --------
        - Decode geohash into latitude/longitude.
        - Add weather interaction features.
        - Encode road categories.
        - Generate aggregate statistics.

        Notes
        -----
        This method should:
        1. Work for both train and test data.
        2. Not retrain or recompute mappings.
        3. Return a transformed copy of the dataframe.
        """
        pass