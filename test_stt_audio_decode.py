"""Exercise Whisper's real audio decoder without downloading a model."""

import io
import unittest
import wave

import numpy as np
from faster_whisper.audio import decode_audio


class STTAudioDecodeTests(unittest.TestCase):
    def test_decode_wav_resamples_stereo_to_whisper_mono(self):
        rate = 48000
        signal = (np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 16000).astype(
            "<i2"
        )
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(np.column_stack((signal, signal)).tobytes())
        audio.seek(0)

        decoded = decode_audio(audio, sampling_rate=16000)

        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.shape, (16000,))
        self.assertTrue(np.isfinite(decoded).all())
        self.assertGreater(float(np.max(np.abs(decoded))), 0.1)


if __name__ == "__main__":
    unittest.main()
