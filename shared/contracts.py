"""
Shared data contracts for the When Attention Drops (WAD) pipeline.

These structures define how components exchange data, independently
of their internal implementations.

See docs/interfaces.md for conventions and integration guidelines.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AudioWindow:
    """
    A time-aligned window of lecture audio.

    Produced by the audio input/windowing component and consumed by
    transcription and acoustic analysis.

    Attributes:
        lectureID: Unique identifier for the lecture/session.
        index: Zero-based index of the audio window.
        startSec: Window start time relative to lecture start.
        endSec: Window end time relative to lecture start.
        samples: Mono audio samples as a 1D float32 NumPy array.
        sampleRate: Audio sampling rate in Hz.
    """

    lectureID: str
    index: int
    startSec: float
    endSec: float
    samples: np.ndarray
    sampleRate: int


@dataclass
class TranscriptSegment:
    """
    A finalized, timestamped segment of transcribed speech.

    Produced by transcription and consumed by NLP analysis.

    Transcript segments are independent of audio windows: one window
    may produce multiple segments, and segments may span window
    boundaries.


    Attributes:
        lectureID: Identifier of the source lecture/session.
        index: Zero-based index within the transcript.
        startSec: Segment start time relative to lecture start.
        endSec: Segment end time relative to lecture start.
        text: Transcribed speech.
    """

    lectureID: str
    index: int
    startSec: float
    endSec: float
    text: str


@dataclass
class FeatureWindow:
    """
    Numerical features associated with an audio window.

    Produced by feature extraction components (e.g., NLP and acoustic
    analysis) and consumed by feature aggregation.

    Each producer emits one finalized FeatureWindow per audio window.
    Results may be delayed when additional context is required.

    Different producers can provide different features for the same
    window. Feature aggregation combines them using their shared
    lecture and window identifiers.

    Attributes:
        lectureID: Identifier of the source lecture/session.
        windowIndex: Index of the corresponding AudioWindow.
        features: Mapping of feature names to numerical values.
    """

    lectureID: str
    windowIndex: int
    features: dict[str, float | None]


@dataclass
class TopicBoundary:
    """
    A detected transition between lecture topics.

    Produced by NLP topic segmentation.

    Detection may require contextual information, so a boundary
    can be emitted after its corresponding timestamp.

    Attributes:
        lectureID: Identifier of the source lecture/session.
        timeSec: Timestamp at which a new topic begins, relative
            to the beginning of the lecture.
    """

    lectureID: str
    timeSec: float