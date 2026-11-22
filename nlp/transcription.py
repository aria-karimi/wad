"""
Lecture audio transcription using faster-whisper.
Accepts AudioWindow objects and produces TranscriptSegment objects.
"""

import tomllib
from math import isclose
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad
from shared.contracts import AudioWindow, TranscriptSegment

configPath = Path(__file__).with_name("config.toml")
with configPath.open("rb") as file:
    config = tomllib.load(file)["transcription"]


class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            config["modelName"],
            device=config["device"],
            compute_type=config["computeType"]
        )
        self.vadModel = load_silero_vad(onnx=True)
        self.sampleRate = config["sampleRate"]
        self.silenceSamples = int(config["minSilenceDurationMs"] * self.sampleRate / 1000)
        self.maxBufferSamples = int(config["maxBufferDurationSec"] * self.sampleRate)
        if not 0 < self.silenceSamples < self.maxBufferSamples:
            raise ValueError("Buffer duration must exceed the positive silence duration.")

        self.buffer = np.empty(0, dtype=np.float32)
        self.bufferStartSec = 0.0
        self.lectureID = None
        self.nextIndex = 0
        self.nextWindowStartSec = None

    def process(self, window: AudioWindow) -> list[TranscriptSegment]:
        """Append contiguous audio and transcribe speech followed by a pause."""
        if window.sampleRate != self.sampleRate or window.samples.ndim != 1:
            raise ValueError(f"Audio must be mono and sampled at {self.sampleRate} Hz.")
        if not isclose(
            window.endSec - window.startSec,
            len(window.samples) / self.sampleRate,
            rel_tol=0, abs_tol=1 / self.sampleRate,
        ):
            raise ValueError("Window timestamps must match its sample count.")
        if self.lectureID is not None and window.lectureID != self.lectureID:
            raise ValueError("Create a new Transcriber for each lecture.")
        if self.nextWindowStartSec is not None and not isclose(
            window.startSec, self.nextWindowStartSec,
            rel_tol=0, abs_tol=1 / self.sampleRate,
        ):
            raise ValueError("Audio windows must be contiguous and non-overlapping.")

        if self.lectureID is None:
            self.lectureID = window.lectureID
            self.bufferStartSec = window.startSec
        self.buffer = np.concatenate((self.buffer, window.samples))
        self.nextWindowStartSec = window.endSec
        return self._processBuffer()

    def flush(self) -> list[TranscriptSegment]:
        """Process remaining speech when the lecture ends."""
        return self._processBuffer(final=True)

    def _processBuffer(self, final: bool = False) -> list[TranscriptSegment]:
        results = []
        while self.buffer.size:
            audio = self.buffer[:self.maxBufferSamples]
            speechRegions = get_speech_timestamps(
                audio,
                self.vadModel,
                sampling_rate=self.sampleRate,
                min_speech_duration_ms=0,
                min_silence_duration_ms=config["minSilenceDurationMs"],
                speech_pad_ms=0,
            )

            # VAD also closes unfinished speech at the supplied audio's edge.
            # Require actual trailing silence before accepting that boundary.
            completedEnds = [
                region["end"] + self.silenceSamples
                for region in speechRegions
                if region["end"] + self.silenceSamples <= len(audio)
            ]
            if completedEnds:
                cut = completedEnds[-1]
            elif final or len(audio) == self.maxBufferSamples:
                cut = len(audio)
            else:
                break

            if speechRegions:
                segments, _ = self.model.transcribe(
                    audio[:cut],
                    language=config["language"],
                    beam_size=config["beamSize"],
                )
                # Finish iteration before removing any buffered audio.
                batch = []
                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue
                    batch.append(TranscriptSegment(
                        lectureID=self.lectureID,
                        index=self.nextIndex + len(batch),
                        startSec=self.bufferStartSec + segment.start,
                        endSec=self.bufferStartSec + segment.end,
                        text=text,
                    ))
                results.extend(batch)
                self.nextIndex += len(batch)

            self.buffer = self.buffer[cut:].copy()
            self.bufferStartSec += cut / self.sampleRate
        return results
