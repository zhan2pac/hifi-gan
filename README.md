# Neural Vocoder

## Installation

1. Install dependencies

```bash
pip install -r ./requirements.txt
```

2. Download checkpoints and pre-trained model

```bash
python3 scripts/download_model.py
```

## Training

If you want to reproduce training process run following command.

```bash
pyhton3 train.py -cn=train
```

## Inference

If you want to inference model on custom dataset run following command.

```bash
python3 synthesize.py -cn=synthesize \
datasets.inference.data_dir=PATH_TO_CUSTOM_DATASET \
inferencer.from_pretrained=PATH_TO_MODEL \
inferencer.save_path=SAVE_FOLDER
```

By default `synthesize` will log generated mel spectrograms and synthesized audio to WandB server. Predictions will be additionally saved to the `SAVE_FOLDER`.

Custom dataset at `PATH_TO_CUSTOM_DATASET` should be the directory of the following format:
```bash
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

Note that dataloader uses batch_size=1 for synthesis and resynthesis. 

If you want to resynthesize audio run the following command.
```bash
python3 resynthesize.py -cn=resynthesize \
datasets=DATASET_CONFIG \
inferencer.from_pretrained=PATH_TO_MODEL \
inferencer.save_path=SAVE_FOLDER
```
where `DATASET_CONFIG` can be `samples_val` for generating LJ-Speech samples or `samples_test` for generating out of sample. You should add desired samples to `data/LJSpeech-1.1/wavs` and `test.txt` file to `data/LJSpeech-1.1` with the names of added audio.

Example of `test.txt`:
```
dla|...transcription dla...
shostakovich|...transcription shostakovich...
theremin|...transcription theremin...
pupin|...transcription pupin...
bernstein|...transcription bernstein...
```


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)