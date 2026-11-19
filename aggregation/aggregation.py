import numpy as np
import pandas as pd
from typing import Iterable

from shared.contracts import AudioWindow, FeatureWindow, TopicBoundary

class AggregationError(ValueError):
    """Raised when an aggregation operation fails due to invalid input data."""


BASE_COLUMNS = [ #these are the columns and they shouldnt be the same as any feature names
    "lectureID",
    "windowIndex",
    "startSec",
    "endSec",
    "topicID",
    "isTopicStart",
    "secSinceTopicStart",
    "secSinceLectureStart",
]
   
def checkWindows(windows: list[AudioWindow]) -> None:
        """
        window indices are 0..N-1 without gaps.
        Raise AggregationError if not.
        """
        for i, window in enumerate(windows):
            if window.index != i:
                raise AggregationError(f"Window {i} has windowIndex {window.index}, expected {i}")
        

class FeatureAggregator:
    """
    Collects audio windows, features and topic boundaries for ONE lecture
    as they arrive, and builds the time-aligned feature table on request.
    """

    def __init__(self, lectureID: str):
        self.lectureID = lectureID
        self.windows: dict[int, AudioWindow] = {}          # windowIndex -> AudioWindow
        self.merged: dict[int, dict[str, float]] = {}      # windowIndex -> {featureName: value} 
        self.topicStarts: set[float] = set()               # boundary times

    def addWindow(self, window: AudioWindow) -> None:
        if window.lectureID != self.lectureID:
            raise AggregationError(f"Window has lectureID {window.lectureID}, expected {self.lectureID}")
        if window.index in self.windows:
            raise AggregationError(f"Window {window.index} was added twice.")
        if window.startSec >= window.endSec:
            raise AggregationError(f"Window {window.index} has startSec >= endSec")

        self.windows[window.index] = window      #store the window
        self.merged[window.index] = {}           #create its empty bucket for this window

    def addFeatures(self, fw: FeatureWindow) -> None:
        if fw.lectureID != self.lectureID:
            raise AggregationError(f"FeatureWindow has lectureID {fw.lectureID}, expected {self.lectureID}")
        if fw.windowIndex not in self.merged:
            raise AggregationError(f"FeatureWindow refers to window {fw.windowIndex}, which hasn't been added yet.")

        row = self.merged[fw.windowIndex]
        for featureName, featureValue in fw.features.items():
            if featureName in BASE_COLUMNS:
                raise AggregationError(f"Feature name '{featureName}' is reserved for a context column.")
            # the new value replaces the old one
            row[featureName] = np.nan if featureValue is None else float(featureValue)

    def addBoundaries(self, boundaries: Iterable[TopicBoundary]) -> None:
        for b in boundaries:
            if b.lectureID != self.lectureID:
                raise AggregationError(f"TopicBoundary has lectureID {b.lectureID}, expected {self.lectureID}")
            self.topicStarts.add(b.timeSec)

    def getFeatures(self) -> pd.DataFrame:
        windows = sorted(self.windows.values(), key=lambda w: w.index) #sort them by index so we can check for gaps and build the table in order
        if not windows:
            raise AggregationError("No audio windows given.")
        checkWindows(windows) # gaps can only be checked once all windows are in

        featureNames = sorted({featureName for row in self.merged.values() for featureName in row})

        lectureStart = windows[0].startSec
        topicStartsList = sorted(self.topicStarts | {lectureStart})

        records = []
        for w in windows:
            midpoint = (w.startSec + w.endSec) / 2 #A window must belong to one topic so we use the midpoint to determine which topic it belongs to. 
            topicID = 0
            for i, start in enumerate(topicStartsList):
                if start > midpoint:
                    break
                topicID = i

            topicStart = topicStartsList[topicID]
            isTopicStart = any(w.startSec <= t < w.endSec for t in topicStartsList)

            record = {
                "lectureID": self.lectureID,
                "windowIndex": w.index,
                "startSec": w.startSec,
                "endSec": w.endSec,
                "topicID": topicID,
                "isTopicStart": isTopicStart,
                "secSinceTopicStart": max(0.0, midpoint - topicStart),
                "secSinceLectureStart": midpoint - lectureStart,
            }
            for featureName in featureNames:
                record[featureName] = self.merged[w.index].get(featureName, np.nan)
            records.append(record)

        df = pd.DataFrame.from_records(records, columns=BASE_COLUMNS + featureNames)
        return df.astype({name: "float64" for name in featureNames})

 