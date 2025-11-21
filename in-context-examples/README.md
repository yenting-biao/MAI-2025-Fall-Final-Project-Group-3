# Explanation of In-Context Examples

This directory contains several in-context examples used in our project. Please ignore the Python scripts as they are just used when I generate the examples. 

- `generated_in_context_examples.json`: This file contains the generated in-context examples used for various tasks.
  - 1st-level key: Task name, including `ASR` (Automatic Speech Recognition), `SER` (Speaker Emotion Recognition), `GR` (Gender Recognition), and `MMAU`.
  - 2nd-level key: Constraint type, including `chain-of-thought`, `closed_ended_questions`, and `creative_writing`.
  - 3rd-level key (if any): Specific constraint for the constraint type.
  - Value: A list of in-context examples for the corresponding task and constraint.
    - Each example is represented as a dictionary containing relevant information such as `audio_path`, `instruction`, and `answer`. The `audio_path` points to the audio file used in the example, where you can refer to the `audios/` folder for the actual audio files.
- `audios/`: This folder contains audio files used in the in-context examples. They are sampled from the CREMA-D dataset.
  - `CREMA-D_metadata.csv`: Metadata file for the sampled audio files from the CREMA-D dataset, including information such as `filename`, `gender`, `emotion`, and `transcription`.