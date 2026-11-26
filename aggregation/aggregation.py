import numpy as np
import pandas as pd
from typing import Iterable

from shared.contracts import FeatureWindow, TopicBoundary


class AggregationError(ValueError):
    """Raised when an aggregation operation fails due to invalid input data."""


# The only columns the aggregator creates itself. Feature names can't use these names.
BASE_COLUMNS = [
    "lectureID",
    "windowIndex",
]


class FeatureAggregator:
    """
    Collects feature windows and topic boundaries for ONE lecture
    as they arrive, and builds the feature table on request.
    """

    def __init__(self, lectureID: str):
        self.lectureID = lectureID
        self.merged: dict[int, dict[str, float]] = {}      # windowIndex -> {featureName: value}
        self.topicBoundaries: list[TopicBoundary] = [] 

    def addFeatures(self, fw: FeatureWindow) -> None:
        if fw.lectureID != self.lectureID:
            raise AggregationError(f"FeatureWindow has lectureID {fw.lectureID}, expected {self.lectureID}")

        # Get this window's bucket. If the window is new, create an empty bucket first 
        row = self.merged.setdefault(fw.windowIndex, {})

        for featureName, featureValue in fw.features.items():
            if featureName in BASE_COLUMNS:
                raise AggregationError(f"Feature name '{featureName}' is reserved.")

            # None means "no value". It never overwrites an existing value.
            if featureValue is None:
                if featureName not in row:
                    row[featureName] = np.nan  # first time we see this feature: store it as missing
                continue                       # already has a value: leave it unchanged

            # Name already in the bucket so value is updated
            # Name not in the bucket yet so it is added as a new feature 
            row[featureName] = float(featureValue)

    def addBoundaries(self, boundaries: Iterable[TopicBoundary]) -> None:
        boundaries = list(boundaries)
        # Check the whole list first, so a bad list never replaces the good one.
        for b in boundaries:
            if b.lectureID != self.lectureID:
                raise AggregationError(f"TopicBoundary has lectureID {b.lectureID}, expected {self.lectureID}")
        self.topicBoundaries = boundaries  # every new list replaces the old one

    def getFeatures(self) -> pd.DataFrame:
        if not self.merged:
            raise AggregationError("No features received yet.")

        # Every feature name that arrived for any window becomes a column.
        featureNames = sorted({featureName for row in self.merged.values() for featureName in row})

        records = []
        for windowIndex in sorted(self.merged):  # sorting a dict sorts its keys -> rows in window order
            record = {"lectureID": self.lectureID, "windowIndex": windowIndex}
            for featureName in featureNames:
                # Features this window never received become NaN.
                record[featureName] = self.merged[windowIndex].get(featureName, np.nan)
            records.append(record)

        df = pd.DataFrame.from_records(records, columns=BASE_COLUMNS + featureNames)
        return df.astype({name: "float64" for name in featureNames})