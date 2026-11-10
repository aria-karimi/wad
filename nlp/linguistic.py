"""
Linguistic feature extraction.
Extracts numerical linguistic features from transcribed speech.
"""

import spacy
import tomllib
from math import isclose, isfinite
from pathlib import Path
from spacy.cli import download
from silero_vad import get_speech_timestamps, load_silero_vad
from shared.contracts import AudioWindow, TranscriptSegment, FeatureWindow

configPath = Path(__file__).with_name("config.toml")
with configPath.open("rb") as file:
    config = tomllib.load(file)["linguistic"]

CONTENT_WORD_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FEATURE_NAMES = {
    "speakingPace", "averageSentenceLength", "questionCount",
    "lexicalNovelty", "lexicalDensity", "fillerTokenCount", "nonSpeechFraction"
}


class LinguisticAnalyzer:
    """Analyze one lecture, identified by the first valid segment or window."""

    def __init__(self):
        self.lectureID = None
        self.historyWindowCount = config["historyWindowCount"]
        if self.historyWindowCount < 1:
            raise ValueError("historyWindowCount must be positive.")
        self.enabledFeatures = list(config["enabledFeatures"])
        if set(self.enabledFeatures) - FEATURE_NAMES:
            raise ValueError("enabledFeatures contains one or more unknown linguistic feature.")
        self.fillerTokenForms = {form.lower() for form in config["fillerTokenForms"]}
        modelName = config["modelName"]
        if not spacy.util.is_package(modelName) and not Path(modelName).exists():
            print(f"Downloading spaCy model {modelName}...", flush=True)
            download(modelName)
        self.spacyModel = spacy.load(modelName, disable=["ner"])
        self.vadModel = None
        if "nonSpeechFraction" in self.enabledFeatures:
            self.vadThreshold = config["vadThreshold"]
            self.vadMinSilenceDurationMs = config["vadMinSilenceDurationMs"]
            if not 0 <= self.vadThreshold <= 1:
                raise ValueError("vadThreshold must be between 0 and 1.")
            if self.vadMinSilenceDurationMs < 0:
                raise ValueError("vadMinSilenceDurationMs must be non-negative.")
            self.vadModel = load_silero_vad(onnx=True)
        self.transcript: list[TranscriptSegment] = []
        self.sentences: list[TranscriptSegment] = []
        self.segmentStatistics: dict[int, dict] = {}
        self.sentenceStatistics: dict[int, dict] = {}
        self.windowLemmas: dict[int, set[str]] = {}
        self.pendingText = ""
        # Character ranges connect buffered text to its source segments.
        self.pendingSegments: list[tuple[int, int, TranscriptSegment]] = []

    def extractFeatures(self, audioWindow: AudioWindow) -> FeatureWindow:
        """Extract window features; text uses midpoints and VAD uses audio durations.
        The caller controls window finalization.
        Repeating a request replaces that window's lemma set rather than adding history.
        """
        startSec = audioWindow.startSec
        endSec = audioWindow.endSec
        windowIndex = audioWindow.index
        if type(windowIndex) is not int or windowIndex < 0:
            raise ValueError("Window index must be a non-negative integer.")
        if not (isfinite(startSec) and isfinite(endSec) and 0 <= startSec < endSec):
            raise ValueError("Window times must be finite and satisfy 0 <= startSec < endSec.")
        self._checkLectureID(audioWindow.lectureID)
        nonSpeechFraction = (
            self._nonSpeechFraction(audioWindow)
            if "nonSpeechFraction" in self.enabledFeatures else None
        )
        self.lectureID = audioWindow.lectureID
        relevantSegments = [
            segment for segment in self.transcript
            if startSec <= (segment.startSec + segment.endSec) / 2 < endSec
        ]
        wordCount = 0
        currentLemmas = set()
        eligibleTokenCount = 0
        contentTokenCount = 0
        fillerTokenCount = 0
        for segment in relevantSegments:
            statistics = self.segmentStatistics[segment.index]
            wordCount += statistics["wordCount"]
            currentLemmas.update(statistics["lemmas"])
            eligibleTokenCount += statistics["eligibleTokenCount"]
            contentTokenCount += statistics["contentTokenCount"]
            fillerTokenCount += statistics["fillerTokenCount"]
        self.windowLemmas[windowIndex] = currentLemmas
        features = {
            "speakingPace": 60 * wordCount / (endSec - startSec),
            "averageSentenceLength": self._averageSentenceLength(startSec, endSec),
            "questionCount": self._questionCount(startSec, endSec),
            "lexicalNovelty": self._lexicalNovelty(windowIndex, currentLemmas),
            "lexicalDensity": self._lexicalDensity(contentTokenCount, eligibleTokenCount),
            "fillerTokenCount": fillerTokenCount,
            "nonSpeechFraction": nonSpeechFraction
        }
        return FeatureWindow(
            lectureID=self.lectureID,
            windowIndex=windowIndex,
            features={name: features[name] for name in self.enabledFeatures}
        )

    def accumulateTranscript(self, segment: TranscriptSegment):
        """Append a transcript segment to the internal transcript."""
        self._checkLectureID(segment.lectureID)
        if self.transcript and segment.index != self.transcript[-1].index + 1:
            raise ValueError("Transcript segments must be contiguous.")
        statistics = self._segmentStatistics(segment)
        self.lectureID = segment.lectureID
        self.transcript.append(segment)
        self.segmentStatistics[segment.index] = statistics

        text = segment.text.strip()
        if text:
            if self.pendingText:
                self.pendingText += " "
            startChar = len(self.pendingText)
            self.pendingText += text
            self.pendingSegments.append((startChar, len(self.pendingText), segment))
            self._extractSentences()

    def flush(self):
        """Treat the remaining text as final when the lecture ends."""
        self._extractSentences(final=True)

    def _checkLectureID(self, lectureID: str):
        """Validate an input's lecture ID before binding it or changing state."""
        if not isinstance(lectureID, str) or not lectureID.strip():
            raise ValueError("A non-empty lectureID is required.")
        if self.lectureID is not None and lectureID != self.lectureID:
            raise ValueError("Create a new LinguisticAnalyzer for each lecture.")

    def _extractSentences(self, final: bool = False):
        """Save completed sentences and retain the last one for more context."""
        if not self.pendingText:
            return
        doc = self.spacyModel(self.pendingText)
        sentences = list(doc.sents)
        if not sentences:
            return
        completedSentences = sentences if final else sentences[:-1]
        for sentence in completedSentences:
            sourceSegments = [
                segment for startChar, endChar, segment in self.pendingSegments
                if startChar < sentence.end_char and endChar > sentence.start_char
            ]
            if not sourceSegments:
                continue
            wordCount = self._countWords(sentence)
            if not wordCount:
                continue
            sentenceIndex = len(self.sentences)
            self.sentences.append(TranscriptSegment(
                lectureID=self.lectureID,
                index=sentenceIndex,
                startSec=sourceSegments[0].startSec,
                endSec=sourceSegments[-1].endSec,
                text=sentence.text.strip()
            ))
            # Ignore closing quotation marks/brackets when looking for a final '?'.
            ending = sentence.text.rstrip(" \t\r\n\"'’”»)]}")
            self.sentenceStatistics[sentenceIndex] = {
                "wordCount": wordCount,
                "isQuestion": ending.endswith("?")
            }

        # Retain the final sentence, including any partial source segment.
        cut = len(self.pendingText) if final else sentences[-1].start_char
        self.pendingText = self.pendingText[cut:]
        self.pendingSegments = [
            (max(0, startChar - cut), endChar - cut, segment)
            for startChar, endChar, segment in self.pendingSegments
            if endChar > cut
        ]

    def _averageSentenceLength(self, startSec: float, endSec: float) -> float | None:
        """Average written-word count for sentence midpoints in [startSec, endSec)."""
        sentenceLengths = [
            self.sentenceStatistics[sentence.index]["wordCount"]
            for sentence in self.sentences
            if startSec <= (sentence.startSec + sentence.endSec) / 2 < endSec
        ]
        if not sentenceLengths:
            return None
        return sum(sentenceLengths) / len(sentenceLengths)

    def _questionCount(self, startSec: float, endSec: float) -> int:
        """Count completed questions assigned by their sentence midpoints."""
        return sum(
            self.sentenceStatistics[sentence.index]["isQuestion"]
            for sentence in self.sentences
            if startSec <= (sentence.startSec + sentence.endSec) / 2 < endSec
        )

    def _lexicalNovelty(self, windowIndex: int, currentLemmas: set[str]) -> float | None:
        """Fraction of content lemmas absent from the preceding N audio windows."""
        if not currentLemmas or windowIndex == 0:
            return None
        historyLemmas = set()
        for index in range(max(0, windowIndex - self.historyWindowCount), windowIndex):
            if index not in self.windowLemmas:
                return None  # An unrequested window is not known silence.
            historyLemmas.update(self.windowLemmas[index])
        if not historyLemmas:
            return None
        return len(currentLemmas - historyLemmas) / len(currentLemmas)

    def _lexicalDensity(self, contentTokenCount: int, eligibleTokenCount: int) -> float | None:
        """Content-POS tokens divided by all alphabetic tokens, including function words."""
        if not eligibleTokenCount:
            return None
        return contentTokenCount / eligibleTokenCount

    def _nonSpeechFraction(self, audioWindow: AudioWindow) -> float | None:
        """Fraction of window samples classified as non-speech, including edge silence.

        Run VAD independently per window. Short gaps can remain inside speech
        intervals according to vadMinSilenceDurationMs. Missing samples return None.
        """
        samples = audioWindow.samples
        if audioWindow.sampleRate != 16000 or samples.ndim != 1:
            raise ValueError("VAD audio must be mono and sampled at 16000 Hz.")
        if not len(samples):
            return None
        if not isclose(
            len(samples) / audioWindow.sampleRate,
            audioWindow.endSec - audioWindow.startSec,
            rel_tol=0, abs_tol=1 / audioWindow.sampleRate,
        ):
            raise ValueError("Window timestamps must match its sample count.")
        speechRegions = get_speech_timestamps(
            samples,
            self.vadModel,
            sampling_rate=audioWindow.sampleRate,
            threshold=self.vadThreshold,
            min_speech_duration_ms=0,
            min_silence_duration_ms=self.vadMinSilenceDurationMs,
            speech_pad_ms=0,
        )
        # Silero returns non-overlapping sample ranges; no padding is added here.
        speechSamples = sum(region["end"] - region["start"] for region in speechRegions)
        return 1 - speechSamples / len(samples)

    def _segmentStatistics(self, segment: TranscriptSegment) -> dict:
        """Cache token and lemma statistics for one incoming segment."""
        doc = self.spacyModel(segment.text)
        eligibleTokens = [token for token in doc if token.is_alpha]
        contentTokens = [token for token in eligibleTokens if token.pos_ in CONTENT_WORD_POS]
        return {
            "wordCount": self._countWords(doc),
            "lemmas": {
                token.lemma_.lower() for token in contentTokens
                if token.lower_ not in self.fillerTokenForms
            },
            "eligibleTokenCount": len(eligibleTokens),
            "contentTokenCount": len(contentTokens),
            "fillerTokenCount": self._fillerTokenCount(doc)
        }

    def _fillerTokenCount(self, doc) -> int:
        """Count configured forms within a segment, excluding uh-huh and uh-oh."""
        tokens = [token for token in doc if not token.is_space]
        hyphens = {"-", "‐", "‑"}
        count = 0
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.lower_ == "uh":
                following = index + 1
                if following < len(tokens) and tokens[following].text in hyphens:
                    following += 1
                if following < len(tokens) and tokens[following].lower_ in {"huh", "oh"}:
                    index = following + 1
                    continue
            if token.lower_ in self.fillerTokenForms:
                count += 1
            index += 1
        return count

    def _wordCount(self, segment: TranscriptSegment) -> int:
        """Count written words using the English joining rules below."""
        doc = self.spacyModel.make_doc(segment.text)
        return self._countWords(doc)

    def _countWords(self, doc) -> int:
        """Count written words in a spaCy Doc or sentence Span.

        Contractions and hyphenated compounds count as one written word.
        En dashes (U+2013) and em dashes (U+2014) remain word boundaries.
        """
        wordCount = 0
        previousEnd = -1
        joiners = {
            "'",  # U+0027 APOSTROPHE
            "’",  # U+2019 RIGHT SINGLE QUOTATION MARK
            "ʼ",  # U+02BC MODIFIER LETTER APOSTROPHE
            "-",  # U+002D HYPHEN-MINUS
            "‐",  # U+2010 HYPHEN
            "‑",  # U+2011 NON-BREAKING HYPHEN
        }
        for token in doc:
            # Preserve an internal apostrophe or hyphen as part of the word.
            if token.text in joiners and token.idx == previousEnd:
                previousEnd = token.idx + len(token)
                continue
            # Ignore tokens without letters or numbers.
            if not any(character.isalnum() for character in token.text):
                previousEnd = -1
                continue
            # Adjacent fragments such as "can" + "not" count together.
            if token.idx != previousEnd:
                wordCount += 1
            previousEnd = token.idx + len(token)

        return wordCount

    def _speakingPace(self, segment: TranscriptSegment) -> float:
        """Return the speaking pace in written words per minute."""
        duration = segment.endSec - segment.startSec
        if duration <= 0:
            raise ValueError("Segment duration must be positive.")
        return 60 * self._wordCount(segment) / duration
