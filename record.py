"""Record a test audio clip from the microphone."""

import argparse
import sys
from pathlib import Path

import sounddevice as sd
import scipy.io.wavfile as wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--duration", type=float, default=10.0)
    parser.add_argument("-o", "--output", default="test_audio.wav")
    args = parser.parse_args()

    if Path(args.output).exists():
        print(f"Error: {args.output} already exists. Delete it first.")
        sys.exit(1)

    import numpy as np
    print(f"Recording {args.duration}s... speak now")
    audio = sd.rec(int(args.duration * 16000), samplerate=16000,
                   channels=1, dtype="float32")
    sd.wait()
    int16 = (audio[:, 0] * 32767).clip(-32768, 32767).astype(np.int16)
    wav.write(args.output, 16000, int16)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
