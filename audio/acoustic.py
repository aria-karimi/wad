"""Bir ses penceresinden duraklama, perde ve ses şiddeti ölçümleri çıkarır."""

import numpy as np
import parselmouth
from silero_vad import get_speech_timestamps, load_silero_vad

from shared.contracts import AudioWindow, FeatureWindow


class AcousticAnalyzer:
    def __init__(self):
        # Hazır VAD modelini bir kez yükler, bütün pencerelerde tekrar kullanırız.
        self.vadModel = load_silero_vad()

    def extractFeatures(self, window: AudioWindow) -> FeatureWindow:
        # AudioInput'ın ürettiği mono, 16 kHz sesle çalışırız.
        if window.sampleRate != 16000 or window.samples.ndim != 1:
            raise ValueError("Ses tek kanallı ve 16000 Hz olmalı.")
        if window.samples.size == 0 or not np.isfinite(window.samples).all():
            raise ValueError("Ses penceresi boş veya geçersiz örnekler içeriyor.")

        # VAD, konuşma aralıklarının başlangıç ve bitiş örneklerini verir.
        # En az 500 ms sessizlik iki konuşma aralığını ayırır; kenarlara dolgu eklemeyiz.
        speechRegions = get_speech_timestamps(
            window.samples,
            self.vadModel,
            sampling_rate=window.sampleRate,
            min_silence_duration_ms=500,
            speech_pad_ms=0,
        )

        # Yalnızca bu pencerenin içindeki iki konuşma aralığı arasındaki boşlukları sayarız.
        # Pencerenin başındaki ve sonundaki sessizliği duraklama olarak saymayız.
        pauseCount = float(max(0, len(speechRegions) - 1))
        pitchValues = []
        loudnessValues = []

        for region in speechRegions:
            # Perde ve ses şiddetini, VAD'nin konuşma olarak belirlediği seslerden ölçeriz.
            speech = window.samples[region["start"]:region["end"]]
            if len(speech) / window.sampleRate < 0.064:
                continue  # Çok kısa ses parçaları güvenilir bir Praat ölçümü vermez.
            sound = parselmouth.Sound(speech, sampling_frequency=window.sampleRate)

            # Hz cinsinden perde değerlerini alırız. Sıfır, perde ölçülemediği anlamına gelir.
            pitch = sound.to_pitch().selected_array["frequency"]
            pitch = pitch[(pitch > 0) & np.isfinite(pitch)]
            pitchValues.extend(pitch)

            # Konuşmanın ses şiddetini dB cinsinden alır, geçersiz değerleri eleriz.
            loudness = sound.to_intensity().values[0]
            loudnessValues.extend(loudness[np.isfinite(loudness)])

        # Standart sapma, ölçümlerin pencere boyunca ne kadar değiştiğini gösterir.
        # Ölçüm yoksa None kullanırız; böylece eksik değer sıfırla karışmaz.
        pitchVariation = None
        loudnessVariation = None
        if pitchValues:
            pitchVariation = float(np.std(pitchValues))
        if loudnessValues:
            loudnessVariation = float(np.std(loudnessValues))

        # Aynı ders ve pencere kimlikleriyle sonuçları FeatureAggregator'a teslim ederiz.
        # speakingPace ve nonSpeechFraction dil analizi tarafında zaten hesaplanıyor.
        return FeatureWindow(
            lectureID=window.lectureID,
            windowIndex=window.index,
            features={
                "pauseCount": pauseCount,
                "pitchVariation": pitchVariation,
                "loudnessVariation": loudnessVariation,
            },
        )
