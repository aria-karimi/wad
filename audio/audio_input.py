"""Kayıtlı ders sesini, sırayla AudioWindow nesneleri olarak okur."""

from math import isfinite

import numpy as np
import parselmouth
import soundfile as sf

from shared.contracts import AudioWindow


class AudioInput:
    def __init__(self, lectureID: str, windowDuration: float = 10.0):
        # Süreyi örnek sayısına çevirebilmek için pozitif, sonlu bir değer gerekir.
        if not isfinite(windowDuration) or round(windowDuration * 16000) < 1:
            raise ValueError("Pencere süresi pozitif olmalı ve en az bir örnek içermeli.")

        self.lectureID = lectureID
        self.windowDuration = windowDuration
        self.sampleRate = 16000  # Transkripsiyonun beklediği saniyelik örnek sayısı.
        self.filePath = None
        self._file = None  # Açık ses dosyasını burada tutarız.
        self._windowSamples = round(windowDuration * self.sampleRate)
        self._position = 0  # Şimdiye kadar üretilen 16 kHz ses örneği sayısı.
        self._index = 0

    def open(self, filePath: str) -> None:
        # Önceki dosya açıksa kapatır, yeni dosyayı okumaya hazırlarız.
        self.close()
        self._file = sf.SoundFile(filePath)
        self.filePath = filePath
        self._position = 0
        self._index = 0

    def nextWindow(self) -> AudioWindow | None:
        if self._file is None:
            raise RuntimeError("Önce open(filePath) ile ses dosyasını açmalısın.")

        # Dosyanın toplam süresinin 16 kHz'de kaç örnek edeceğini hesaplarız.
        totalSamples = round(self._file.frames * self.sampleRate / self._file.samplerate)
        sampleCount = min(self._windowSamples, totalSamples - self._position)
        if sampleCount <= 0:
            return None  # Dosya bitti; sıradaki pencere yok.

        # Hedef pencerenin bitişini dosyanın kendi örnekleme hızına çeviririz.
        # Her sınırı toplam konumdan hesaplamak, yuvarlama hatasının birikmesini önler.
        endFrame = round((self._position + sampleCount) * self._file.samplerate / self.sampleRate)
        endFrame = min(endFrame, self._file.frames)
        frameCount = endFrame - self._file.tell()
        if frameCount < 1:
            raise ValueError("Pencere süresi bu dosyanın örnekleme hızı için çok kısa.")

        # Yalnızca bu pencerenin sesini okuruz. Her satır bir anı, sütunlar kanalları taşır.
        channels = self._file.read(frameCount, dtype="float32", always_2d=True)
        samples = channels.mean(axis=1)  # Sol ve sağ kanalı ortalayarak mono yaparız.

        # Gerekirse mevcut Parselmouth kütüphanesiyle örnekleme hızını 16 kHz'e çeviririz.
        if self._file.samplerate != self.sampleRate:
            sound = parselmouth.Sound(samples, sampling_frequency=self._file.samplerate)
            samples = sound.resample(self.sampleRate).values[0].astype(np.float32)

        # Dönüşümde oluşabilen bir örneklik yuvarlama farkını hedef uzunluğa eşitleriz.
        if len(samples) < sampleCount:
            samples = np.pad(samples, (0, sampleCount - len(samples)), mode="edge")
        samples = samples[:sampleCount]

        # Kimlik ve zaman bilgileri, diğer bileşenlerin bu parçayı eşleştirmesini sağlar.
        window = AudioWindow(
            lectureID=self.lectureID,
            index=self._index,
            startSec=self._position / self.sampleRate,
            endSec=(self._position + sampleCount) / self.sampleRate,
            samples=samples,
            sampleRate=self.sampleRate,
        )
        self._position += sampleCount  # Sonraki çağrı bu pencerenin bittiği yerden başlar.
        self._index += 1
        return window  # Dosyanın son parçası kısaysa gerçek uzunluğuyla döner.

    def close(self) -> None:
        # İşimiz bitince işletim sisteminin açık tuttuğu dosyayı serbest bırakırız.
        if self._file is not None:
            self._file.close()
            self._file = None
