# MAI-2025-Fall-Final-Project-Group-3

## Directory format

* [audio_caption](./audio_caption)
  * [captioner.py](./audio_caption/captioner.py)
* [samples](./samples)
  * [sd-qa_1008642825401516622.wav](./samples/sd-qa_1008642825401516622.wav)
  * [sd-rq_6426446469024899068.wav](./samples/sd-rq_6426446469024899068.wav)
* [models](./models)
  * [basemodel.py](./models/basemodel.py)
  * [diva.py](./models/diva.py)
* [smoke_test_diva.py](./smoke_test_diva.py)
* [README.md](./README.md)
* [requirements](./requirements)
     * [diva.txt](./requirements/diva.txt)

- Tool for printing directory tree

    ```bash
    cd /path/to/MAI-2025-Fall-Final-Project-Group-3
    ```

    ```bash
    tree=$(tree -tf --noreport -I '*~' --charset ascii $1 |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g')
    printf "# Project tree\n\n${tree}"
    ```

### How to test

#### DiVA

Create and enter the conda env for running DiVA.

```bash
conda create --name diva python=3.12.12 -y
conda activate diva
pip install -r requirements/diva.txt
```

Run the following command (about 23.5GB VRAM required):

```bash
python smoke_test.py --model diva
```

#### BLSP-Emo

Create and enter the conda env for running DiVA.

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

#### SALMONN

Create and enter the conda env for running SALMONN.

```bash
conda create --name salmonn python=3.10 -y
conda activate salmonn
pip install -r requirements/salmonn.txt
```

Run the following command (about  VRAM required):

```bash
python models/SALMONN/cli_inference.py --
```

