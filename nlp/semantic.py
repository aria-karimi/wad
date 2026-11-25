from shared.contracts import AudioWindow, FeatureWindow, TranscriptSegment
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import tomllib

configPath = Path(__file__).with_name("config.toml")
with configPath.open("rb") as file:
    config = tomllib.load(file)["Semantic"]


class SemanticAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer(config["modelName"])
        self.historyWindowCount = config["historyWindowCount"]
        self.enabledFeatures = config["enabledFeatures"]
        self.lectureID = None
        self.transcript: list[TranscriptSegment] = []
        self.windowEmbeddings: dict[int, np.ndarray | None] = {}

    def accumulateTranscript(self, segment: TranscriptSegment):
        """Accumulate incoming transcript segments."""
        self._checkLectureID(segment.lectureID)
        self.transcript.append(segment)

    def extractFeatures(self, audioWindow: AudioWindow) -> FeatureWindow:
        """Extract semantic features for an audio window."""
        self._checkLectureID(audioWindow.lectureID)
        windowIndex = audioWindow.index
        text = " ".join(
            segment.text for segment in self.transcript
            if audioWindow.startSec
            <= (segment.startSec + segment.endSec) / 2
            < audioWindow.endSec
        ).strip()
        embedding = (
            self.model.encode(text, normalize_embeddings=True)
            if text else None
        )
        self.windowEmbeddings[windowIndex] = embedding
        features = {
            "semanticShift": self._semanticShift(windowIndex)
        }
        return FeatureWindow(
            lectureID=audioWindow.lectureID,
            windowIndex=windowIndex,
            features={
                name: features[name]
                for name in self.enabledFeatures
            }
        )

    def _semanticShift(self, windowIndex: int) -> float | None:
        """Cosine distance from the preceding semantic context."""
        currentEmbedding = self.windowEmbeddings[windowIndex]
        if currentEmbedding is None:
            return None
        history = []
        for index in range(
            max(0, windowIndex - self.historyWindowCount),
            windowIndex
        ):
            if index not in self.windowEmbeddings:
                return None
            embedding = self.windowEmbeddings[index]
            if embedding is not None:
                history.append(embedding)
        if not history:
            return None
        context = np.mean(history, axis=0)
        contextNorm = np.linalg.norm(context)
        if contextNorm == 0:
            return None
        similarity = np.dot(currentEmbedding, context) / contextNorm
        return float(1 - np.clip(similarity, -1, 1))

    def _checkLectureID(self, lectureID: str):
        """Prevent mixing data from different lectures."""
        if self.lectureID is None:
            self.lectureID = lectureID
        elif self.lectureID != lectureID:
            raise ValueError("Create a new SemanticAnalyzer for each lecture.")