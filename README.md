# MAI-2025-Fall-Final-Project-Group-3

### Download test data 
To reproduce our experiment, please refer to the README file of https://github.com/kehanlu/Speech-IFEval/tree/main.
Extract the `data/` directory from that repository and place it in `/path/to/MAI-2025-Fall-Final-Project-Group-3/data`.

### Environment setup
The four models need four different environments, please set up your virtual environments properly.

#### DeSTA2.5-Audio
For DeSTA2_5, we need to download the model manually.
```bash
cd models/
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
cd DeSTA2.5-Audio
pip install -e .
cd ..
```

#### Gemini 3 Pro

To run the experiments involving Gemini 3 Pro, the `GEMINI_API_KEYS` (note the "S")
environment variable needs to be set beforehand with Gemini API keys.
The format of the variable should be as follows:
```bash
GEMINI_API_KEYS='<key1>,<key2>,...,<keyN>'
```
Multiple API keys can be set by delimiting them with commas, but do not add a
trailing comma. Obtain your own API keys from [Google AI Studio](https://aistudio.google.com/).

The environment variable can be set by using the `export` command or by creating
a file called `.env` to the root of the project and adding the variable
to the file. In the latter case, [`python-dotenv`](https://github.com/theskumar/python-dotenv)
is used to extract the variable from the file. The file should look like this:
```bash
GEMINI_API_KEYS='<key1>,<key2>,...,<keyN>'
```

#### All models need to activate their own environment
```bash
conda create --name <your_env_name> python=3.11.2 -y
conda activate <your_env_name>
pip install -r requirements/<model_name>.txt
conda install -c conda-forge ffmpeg
```

### To reproduce our experiments
The default results will be saved in `model_responses/<MODELNAME>`
```bash
bash scripts/<MODELNAME>_ceq.sh
bash scripts/<MODELNAME>_cw.sh
bash scripts/<MODELNAME>_CoT.sh
```
- `<MODELNAME>` : [`qwen`, `qwen2`, `desta2_5`, `blsp_emo`] 

### How to do ICL on assigned IF task and audio task 

```bash
python run.py --model_name <MODELNAME> --audio_task <AUDIOTASK> --response_task <RESPONSETASK> --IF_task <IFTASK> --examples <EXAMPLES> --output_dir <DIR>
```
- `<MODELNAME>` : [`qwen`, `qwen2`, `desta2_5`, `blsp-emo`]
- `<AUDIOTASK>` : [`ASR`, `SER`, `GR`] 
- `<RESPONSETASK>` : [`closed_ended_questions`, `chain-of-thought`, `creative_writing`]
- `<IFTASK>` : Depends on the `<RESPONSETASK>`
- `<EXAMPLES>` : [`0`~`8`]
- `<DIR>` : The output directory you want to save in 

### Default responses format 
The query with metadata (e.g., instruction, the entire messages) and the model responses would be stored in the following dir structure.

```
- model_response/
  - <model_name>
    - ASR/
      - chain-of-thought/
        - chain-of-thought/
          - outout_{examples}-shot_{timestamp}.jsonl
          - ...jsonl
      - closed_ended_questions/
        - ...
      - creative_writing/
        ...
    - GR/
      - chain-of-thought/
        - chain-of-thought/
      - closed_ended_questions/
        - ...
      - creative_writing/
        ...
    - SER/
      - chain-of-thought/
        - chain-of-thought/
      - closed_ended_questions/
        - ...
      - creative_writing/
        ...
```

### Evaluation 

- `<model_name>` : [`qwen`, `qwen2`, `blsp_emo`, `desta2_5`, `qwen25_omni`]

#### Env

```bash
conda activate <your-env>
pip install -r eval_scripts/requirements.txt
```

#### `closed_ended_questions`

```bash
conda activate <your-env>
bash eval_scripts/<model_name>_ceq.sh
```

#### `creative_writing`

```bash
conda activate <your-env>
bash eval_scripts/<model_name>_cw.sh
```

#### `chain-of-thought`

```bash
conda activate <your-env>
bash eval_scripts/<model_name>_cot.sh # evaluate CoT IF rate
bash eval_scripts/<model_name>_cot_task_level.sh # evaluate task level performance of CoT (extract output first then use wer and accuracy)
```

