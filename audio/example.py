"""VS Code'da doğrudan çalıştırılabilir; varsayılan olarak wad/test.wav kullanır."""

import argparse
from pathlib import Path
import sys
from uuid import uuid4

# Bu dosyanın bir üst klasörü, audio ve shared klasörlerini içeren wad klasörüdür.
projectFolder = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    # Normal Run düğmesiyle açıldığında Python'ın proje paketlerini bulmasını sağlarız.
    sys.path.insert(0, str(projectFolder))

from audio.audio_input import AudioInput
from audio.acoustic import AcousticAnalyzer


if __name__ == "__main__":
    # Dosya yolu verilirse onu, normal Run ile açılırsa wad/test.wav kaydını kullanırız.
    parser = argparse.ArgumentParser(description="Ders sesini pencerelere ayırıp akustik özellikleri gösterir.")
    parser.add_argument(
        "filePath", nargs="?", default=str(projectFolder / "test.wav"),
        help="Ses dosyasının yolu; verilmezse wad/test.wav kullanılır.",
    )
    args = parser.parse_args()

    # Her ders için ayrı bir kimlik üretir, 10 saniyelik pencereler kullanırız.
    audioInput = AudioInput(lectureID=uuid4().hex, windowDuration=10.0)
    audioInput.open(args.filePath)
    try:
        analyzer = AcousticAnalyzer()
        window = audioInput.nextWindow()
        while window is not None:
            # Önce ses penceresini, sonra o pencereden çıkan üç ölçümü ekranda görürüz.
            print(f"Pencere {window.index}: {window.startSec:.2f}–{window.endSec:.2f} saniye")
            print(analyzer.extractFeatures(window))
            window = audioInput.nextWindow()
    finally:
        # Okuma veya analiz hata verse bile dosya açık kalmaz.
        audioInput.close()
