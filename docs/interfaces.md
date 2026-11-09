# WAD Pipeline Interfaces

## Overview

This document describes the shared interfaces and conventions used by the When Attention Drops (WAD) pipeline.

Components are developed independently and communicate through the data structures defined in `shared/contracts.py`.

## Data Flow

| Producer | Consumer | Contract |
|---|---|---|
| Audio windowing | Transcription, acoustic analysis | `AudioWindow` |
| Transcription | NLP analysis | `TranscriptSegment` |
| NLP feature extraction | Feature aggregation | `FeatureWindow` |
| Acoustic feature extraction | Feature aggregation | `FeatureWindow` |
| Topic segmentation | Downstream processing | `TopicBoundary` |

Interfaces for model training, prediction, and frontend communication will be defined later.

## Conventions

- All timestamps are in seconds relative to the beginning of the lecture.
- Indices are zero-based.
- Audio windows and transcript segments have independent indices.
- Audio samples are mono, represented as one-dimensional NumPy `float32` arrays.
- Feature windows reference their corresponding audio windows through `lectureID` and `windowIndex`.
- Feature names must be unique across components to avoid collisions.
- Missing feature values are represented by `None`.

## Incremental Processing

The pipeline is designed to support both recorded and live lectures.

- Components must not require access to the complete lecture before processing begins.
- Components may maintain internal state and accumulate context.
- Results may be emitted with a delay when additional context is required.
- Processing state must remain isolated between lectures.