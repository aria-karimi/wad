import numpy as np
import pandas as pd
from typing import Iterable

from shared.contracts import AudioWindow, FeatureWindow, TopicBoundary

class AggregationError(ValueError):
    """Raised when an aggregation operation fails due to invalid input data."""


CONTEXT_COLUMNS = [
    "lectureID",
    "windowIndex",
    "startSec",
    "endSec",
    "topicID",
    "isTopicStart",
    "secSinceTopicStart",
    "secSinceLectureStart",
]

def checkWindows(windows: list[AudioWindow], lectureID: str) -> None:
    """
    Check that all windows have the same lectureID and are sorted by windowIndex.
    Raise AggregationError if not.
    """
    for i, window in enumerate(windows):
        if window.lectureID != lectureID:
            raise AggregationError(f"Window {i} has lectureID {window.lectureID}, expected {lectureID}")
        if window.index != i:
            raise AggregationError(f"Window {i} has windowIndex {window.windowIndex}, expected {i}")
        if window.startSec >= window.endSec:
            raise AggregationError(f"Window {i} has startSec {window.startSec} >= endSec {window.endSec}")

def topicStartsAt(boundaries: Iterable[TopicBoundary],lectureID: str, lectureStart: float) -> list[float]:
    starts = {lectureStart}
    for b in boundaries:
        if b.lectureID != lectureID:
            raise AggregationError(f"TopicBoundary has lectureID {b.lectureID}, expected {lectureID}")
      
        starts.add(b.timeSec)
    return sorted(starts)

def aggregate( 
    windows: Iterable[AudioWindow],
    featureWindows: Iterable[FeatureWindow],
    boundaries: Iterable[TopicBoundary] = (),
) -> pd.DataFrame:
    """
    merged audio windows, feature windows, and topic boundaries for a single lecture into a single DataFrame.
    DataFrame with one row per window, sorted by windowIndex."""

    lectureID = windows[0].lectureID
    checkWindows(windows, lectureID)

    # merge features per window

    validIndices = set()
    for w in windows:
        validIndices.add(w.index)

# keys are windowIndex (integer), values are dicts of features ( feature name(string),numerical value(float))
    merged: dict[int, dict[str, float]] = {} 
    for i in validIndices:
        merged[i] = {} 

    #adding the features from featureWindows to the merged dict
    for fw in featureWindows:
        if fw.lectureID != lectureID:
            raise AggregationError(f"FeatureWindow has lectureID {fw.lectureID}, expected {lectureID}")
        if fw.windowIndex not in validIndices:
            raise AggregationError(f"FeatureWindow has windowIndex {fw.windowIndex}, which isnt a valid index for this lecture. index must be {min(validIndices)}-{max(validIndices)}).")
    
        row= merged[fw.windowIndex]
        for featureName, featureValue in fw.features.items():
            if featureName in row:
              raise AggregationError( f"Feature '{featureName}' sent twice for window {fw.windowIndex}.")
    
            row[featureName] = np.nan if featureValue is None else float(featureValue)

    featureNames = sorted({featureName for row in merged.values() for featureName in row})


    #topic and context columns
    lectureStart = windows[0].startSec
    topicStartsList = topicStartsAt(boundaries, lectureID, lectureStart)

    records = [] # each row is a dictionary
    for w in windows:
        #A window must belong to 1 topic so we take the topic that is active at the midpoint of the window
        midpoint = (w.startSec + w.endSec) / 2 
       
       # topic id is the last topic start that is <= window midpoint
       # if I inserted x into this sorted list, at which position would it go?
        topicID = 0
        for i, start in enumerate(topicStartsList):
            if start > midpoint: 
                break
            topicID = i   #if start <= midpoint, then this is the current topic

        topicStart = topicStartsList[topicID]
        isTopicStart= any(w.startSec <= t < w.endSec for t in topicStartsList)
            
        record = {
                "lectureID": lectureID,
                "windowIndex": w.index,
                "startSec": w.startSec,
                "endSec": w.endSec,
                "topicID": topicID,
                "isTopicStart": isTopicStart,
                "secSinceTopicStart": max(0.0, midpoint - topicStart),
                "secSinceLectureStart": midpoint - lectureStart,
            }
        for featureName in featureNames:
            record[featureName] = merged[w.index].get(featureName, np.nan)
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=CONTEXT_COLUMNS + featureNames)
    return df.astype({name: "float64" for name in featureNames})


        