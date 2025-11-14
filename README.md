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
conda create --name diva python=3.12 -y
conda activate diva
pip install -r requirements/diva.txt
```

Run the following command (about 23.5GB VRAM required):

```bash
bash smoke_test_diva.py
```


