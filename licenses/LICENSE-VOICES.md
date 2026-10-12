# Voice Clip Licenses

## Source & License

All voice clips (except pre-existing ones) are sourced from [OwenTyme/voice-zero](https://github.com/OwenTyme/voice-zero).

**License: CC0 1.0 Universal (Public Domain)**
- No permission required to use, modify, or distribute
- No attribution legally required, but original source credited below
- Original audio from LibriVox, Archive.org, and freesound.org

## Curation Notes

These voices were curated for ezLocalAI's TTS voice cloning system. The model can handle tone/emotion changes natively, so only one reference clip per unique voice is included (no duplicate tone variants).

Selection criteria:
- Recognizable names (famous actors/narrators) preferred
- Wide accent/language diversity (one representative per distinct accent)
- No duplicates of the same person in different tones
- All clips are 5-14 seconds, 24kHz mono s16 PCM WAV

## Pre-existing Voices (not from voice-zero)

| Voice File | Description |
|-----------|-------------|
| DukeNukem.wav | Duke Nukem - classic video game character voice |
| HAL9000.wav | HAL 9000 - 2001: A Space Odyssey computer voice |
| Morgan_Freeman.wav | Morgan Freeman style narrator |
| StarTrekComputer1.wav | Star Trek (TOS) ship computer voice |
| default.wav | Default voice (Duke Nukem) |

## Voice-Zero Voices (CC0)

| # | Voice File | Original Name | Description / Accent |
|---|-----------|---------------|---------------------|
| 1 | american_male_storyteller.wav | Larry Wilson | American male - storytelling/narrative tone (11.8s) |
| 2 | australian_female_casual.wav | Annise | Australian female - casual delivery (6.4s) |
| 3 | australian_male_cultivated.wav | Timothy Ferguson | Australian male - cultivated/formal (7.0s) |
| 4 | bill_mosley.wav | Bill Mosley | Famous actor (The Walking Dead, many films) - deep baritone (11.6s) |
| 5 | british_rp_female.wav | Ruth Golding | British RP female - warm, polished (8.4s) |
| 6 | british_rp_male.wav | Peter Yearsley | British RP male - precise diction, London (5.9s) |
| 7 | french_accented_narrator.wav | Ezwa | French-accented narrator (7.3s) |
| 8 | german_female_narrator.wav | Sonja | German female narrator (7.0s) |
| 9 | hungarian_female_narrator.wav | Diana Majlinger | Hungarian female narrator (6.8s) |
| 10 | jolie_odell.wav | Jolie O'Dell | Professional voice actress/narrator - American standard (10.0s) |
| 11 | midwestern_female_storyteller.wav | Kristin Hughes | Midwestern/Iowa female - storytelling delivery (8.7s) |
| 12 | newfoundland_male.wav | Sean Michael Hogan | Newfoundland, Canada - very distinctive regional accent (6.6s) |
| 13 | robert_middleton.wav | Robert Middleton | Famous narrator (Duck & Cover civil defense film, 1951) (11.2s) |
| 14 | scottish_male_narrator.wav | Ian Skillen | Scottish audiobook narrator - distinctive Scottish accent (6.3s) |
| 15 | singaporean_mandarin_female.wav | Mezzogal | Singaporean Mandarin female (6.9s) |
| 16 | southern_louisiana_female.wav | Pam Castille | Southern Louisiana female - Cajun-influenced (12.0s) |
| 17 | warm_american_narrator.wav | Douglas Harlon | American audiobook narrator - warm tone (9.2s) |

## Technical Format

All voice clips are:
- **Format**: WAV (PCM signed 16-bit)
- **Sample Rate**: 24000 Hz
- **Channels**: Mono (1)
- **Duration**: 5-14 seconds each
- **Purpose**: Reference audio for TTS voice cloning

## Usage

These voices are auto-discovered by the `_local_voice_names()` function in `app.py` which globs `voices/*.wav`. No registry or config file needs updating when adding/removing voices.

The `licenses/` folder is NOT included in the voice glob and does not disrupt voice discovery.
