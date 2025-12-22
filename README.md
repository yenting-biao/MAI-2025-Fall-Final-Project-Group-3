# MAI-2025-Fall-Final-Project-Group-3

## Download test data
To reproduce our experiment, please refer to the README file of https://github.com/kehanlu/Speech-IFEval/tree/main.
Extract the `data/` directory from that repository and place it in `/path/to/MAI-2025-Fall-Final-Project-Group-3/data`.

## Environment setup
The four models need four different environments, please set up your virtual environments properly.

### DeSTA2.5-Audio
For DeSTA2_5, we need to download the model manually.
```bash
cd models/
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
cd DeSTA2.5-Audio
pip install -e .
cd ..
```

Run the following command (about 23.5GB VRAM required):

```bash
python smoke_test.py --model diva
```

#### BLSP-Emo

Create and enter the conda env for running BLSP-Emo.

```bash
conda create --name blsp-emo python=3.11 -y  # Must use python=3.11 as BLSP-Emo uses some old packages like torch==2.0.1 that are not installable in newer Python versions
conda activate blsp-emo
pip install -r models/blsp_emo_package/requirements.txt
pip install pyarrow==12.0.0  # needed to avoid the following error: "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'. Did you mean: 'ExtensionType'?"
```

Create a HuggingFace access token (if you want to use `hf` to download the weights in the next step)

- Create a "fine-grained" token
- Under "Repositories permissions" search for "cwang621/blsp-emo", select it, and check "Read access to contents of selected repos"
- Click "Create token"
- Run `export HF_TOKEN="<your access token>"`

Run the following command (about 20GB VRAM required):

```bash
python smoke_test_blsp_emo.py
```

Note: model weights are automatically downloaded by `blsp_emo.py` using the HuggingFace CLI `hf`. If you want to download the weights manually, you can do so with `hf download cwang621/blsp-emo`; if weights are downloaded manually, remember to specify the path to weights when calling `BLSP_emo()`.


## How to test and run ICL experiments

### Testing data
To download the testing data, please refer to the README file of https://github.com/kehanlu/Speech-IFEval/tree/main

### Environment setup

Create and enter the conda env for running ICL experiments in Qwen, Qwen2 and blsp_emo.

```bash
conda create --name <your_env_name> python=3.11.2 -y
conda activate <your_env_name>
pip install -r requirements/<model_name>.txt
conda install -c conda-forge ffmpeg
```

## To reproduce our experiments
The default results will be saved in `model_responses/<MODELNAME>`
```bash
bash scripts/<MODELNAME>_ceq.sh
bash scripts/<MODELNAME>_cw.sh
bash scripts/<MODELNAME>_CoT.sh
```
- `<MODELNAME>` : [`qwen`, `qwen2`, `desta2_5`, `blsp_emo`]

### How to do ICL on assigned IF task and audio task

```bash
conda create --name <your_env_name> python=3.11.2 -y
conda activate <your_env_name>
pip install -r requirements/DeSTA2_5.txt
conda install -c conda-forge ffmpeg
```
### Arguments

For args details, please refer to the help message of `run.py`:

```bash
python run.py --help
```

### Data
To download the data, please refer to the README file of https://github.com/kehanlu/Speech-IFEval/tree/main


### Test

To test the setup, you can run a smoke test with a small number of examples:

```bash
python run.py --model_name qwen --audio_task ASR --response_task closed_ended_questions --IF_task change_case:english_capital --examples 2 --use_test_sample --verbose --debug
```

### Run ICL experiments

To run In-Context Learning (ICL) experiments, use the following command:

```bash
python run.py --model_name <model_name> --audio_task <audio_task> --response_task <response_task> --IF_task <IF_task> --examples <number_of_examples>
```

Replace `<model_name>`, `<audio_task>`, `<response_task>`, `<IF_task>`, and `<number_of_examples>` with your desired values. For example:

```bash
python run.py --model_name qwen --audio_task ASR --response_task closed_ended_questions --IF_task change_case:english_capital --examples 3
```

## Experiment Suite

### Generate and save LALMs' responses.

```bash
bash scripts/<model_name>_ceq.sh
bash scripts/<model_name>_cw.sh
bash scripts/<model_name>_CoT.sh
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

### Evaluation scripts

- `<model_name>` : [`qwen`, `qwen2`, `blsp_emo`, `desta2_5`]

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

