# Voice Clip Licenses & Attribution

This folder documents the licensing for all voice reference clips used by ezLocalAI's
zero-shot TTS (Coqui XTTS v2 / VITS) pipeline.

## Source Repository

All 20 new voice clips were sourced from:

- **Repository**: [OwenTyme/voice-zero](https://github.com/OwenTyme/voice-zero)
- **License**: [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/public-domain/cc0/)
- **Original sources**: LibriVox.org, Archive.org, freesound.org

Under CC0, the authors have waived all copyright and related rights to the extent
possible under law. No attribution is legally required, but we acknowledge the source
repository and original performers out of respect.

## Voice Clip Inventory

| # | File | Voice / Description | Accent / Origin | Duration |
|---|------|---------------------|-----------------|----------|
| 1 | jolie_odell.wav | Jolie O'Dell – Professional voice actor, clear narration | American (Standard) | 10.0s |
| 2 | bill_mosley.wav | Bill Mosley – Actor (The Walking Dead), deep baritone | American (Metropolitan) | 11.6s |
| 3 | robert_middleton.wav | Robert Middleton – "Duck and Cover" civil defense narrator | American (Ohio) | 11.2s |
| 4 | douglas_harlon.wav | Douglas Harlon – Audiobook narrator, warm tone | American (Standard) | 9.2s |
| 5 | jason_x.wav | Jason X – Energetic male narrator | American (Standard) | 9.8s |
| 6 | ian_skillen.wav | Ian Skillen – Scottish audiobook narrator | Scottish | 6.3s |
| 7 | peter_yearsley.wav | Peter Yearsley – British RP narrator, precise diction | English (London/RP) | 5.9s |
| 8 | ruth_golding.wav | Ruth Golding – British female narrator, warm | English (RP) | 8.4s |
| 9 | ezwa.wav | EZWA – French-accented narrator | French | 7.3s |
| 10 | sonja.wav | Sonja – German female narrator | German | 7.0s |
| 11 | diana_majlinger.wav | Diana Majlinger – Hungarian female narrator | Hungarian | 6.8s |
| 12 | annise.wav | Annise – Australian female, casual tone | Australian (Casual) | 6.4s |
| 13 | timothy_ferguson.wav | Timothy Ferguson – Australian male, cultivated | Australian (Cultivated) | 7.0s |
| 14 | larry_wilson.wav | Larry Wilson – American male, storytelling tone | American (Standard) | 11.8s |
| 15 | kristin_hughes.wav | Kristin Hughes – American female, Iowa accent | American (Midwest/Iowa) | 8.7s |
| 16 | caprisha_page.wav | Caprisha Page – American female, Midwestern | American (Midwestern) | 12.4s |
| 17 | pam_castille.wav | Pam Castille – American female, Southern Louisiana | American (South Louisiana) | 12.0s |
| 18 | sean_michael_hogan.wav | Sean Michael Hogan – Canadian male, Newfoundland accent | Canadian (Newfoundland) | 6.6s |
| 19 | mezzogal.wav | Mezzogal – Chinese female, Singaporean Mandarin | Chinese (Singapore) | 6.9s |
| 20 | greasyplastic-whisper.wav | GreasyPlastic – Whisper/ASMR-style delivery | American (Whisper) | 13.4s |

## Existing Voices (Pre-existing, not from voice-zero)

These voices were already present in ezLocalAI before this addition:

| File | Description | Notes |
|------|-------------|-------|
| Morgan_Freeman.wav | Morgan Freeman-style deep narration | Pre-existing |
| DukeNukem.wav | Duke Nukem game character voice | Pre-existing |
| HAL9000.wav | 2001: A Space Odyssey computer voice | Pre-existing |
| StarTrekComputer1.wav | Original Star Trek (TOS) computer voice | Pre-existing |
| default.wav | Default/fallback TTS voice | Pre-existing |

## License Notes

- **CC0 (Public Domain)**: The user may copy, modify, distribute, and perform the work,
  even for commercial purposes, without asking permission. See full text at
  [creativecommons.org/public-domain/cc0/](https://creativecommons.org/publicdomain/cc0/1.0/legalcode).
- **No warranty**: The work is provided "AS IS" with no warranties of any kind.
- **Star Trek TNG Computer**: A permissively-licensed sample of the Star Trek: The Next
  Generation computer voice was not available. The TOS computer (StarTrekComputer1.wav)
  is already included as a pre-existing asset. Paramount holds copyright on TNG audio;
  no CC0/CC-BY clone was found in public repositories.

## Technical Format

All new clips are converted to:
- **Format**: WAV (PCM signed 16-bit)
- **Sample rate**: 24,000 Hz
- **Channels**: Mono (1)
- **Duration range**: 5.9s – 13.4s (suitable for zero-shot TTS reference)

Conversion tool: FFmpeg (`ffmpeg -i input.flac -ar 24000 -ac 1 -sample_fmt s16 output.wav`)