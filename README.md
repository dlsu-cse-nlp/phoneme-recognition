<div align="center">

<h1>Phoneme recognition of Filipino speech with deep learning methods</h1>

</div>

This serves as the main Git repository for a phoneme recognition undergraduate
research project.

> [!NOTE]
> This README is currently a work in progress.

## Abstract

> Phoneme recognition (PR), the problem of converting spoken language to a
> sequence of phonemes, has seen much progress with the advent of deep learning
> techniques, particularly with high-resource languages. However, in the case
> of Filipino, a low-resource language, existing Filipino-specific PR systems
> are primarily built on classical machine learning architectures that often
> struggle to capture quirks such as stress or glottal stops. Furthermore,
> language-specific Filipino speech processing models are often trained for
> speech-to-grapheme transcription, rather than being trained and evaluated on
> datasets annotated with phonetic labels, due primarily to a shortage of such
> datasets. This study proposes to develop a Filipino-specific PR model with
> deep-learning methods, fine-tuned from existing pre-trained architectures.
> This PR model would be tested against existing Filipino or multilingual PR
> models on phoneme-annotated datasets. The main contributions are thus as
> follows: (1) establishing mixed-source phoneme-annotated datasets augmented by
> a grapheme-to-phoneme (G2P) pipeline; (2) implementing a deep learning-based
> PR model and evaluating its performance on baseline datasets; and (3)
> evaluating its performance on a downstream task, namely mispronunciation
> detection and diagnosis (MD\&D).

## Data directory

**Only the manifest files are tracked by this repository**. I opted not to
use LFS for this as it would quickly hit limits. As such, grab the datasets
listed first, then refer to the following for the intended project directory
structure:

```
data
├── fleurs
│   ├── audio
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   ├── dev.tsv
│   ├── test_manifest.json
│   ├── test.tsv
│   ├── train_manifest.json
│   ├── train.tsv
│   └── valid_manifest.json
├── fsc
│   ├── parquet
│   ├── test_manifest.json
│   ├── train_manifest.json
│   └── valid_manifest.json
└── nexdata
    ├── audio
    │   ├── G00001
    │   └── G00608
    ├── test_manifest.json
    ├── train_manifest.json
    └── valid_manifest.json
```

## COARE setup

We have access to DOST's [Saliksik HPC through COARE](https://coare.asti.dost.gov.ph).
File tree looks something like this:

```
home
├── phoneme-recognition         A clone of this repository.
│                               Kept only as a backup copy. Don't run jobs here.
│                               Has symlinks to the scratch1 clone's data directories.
├── scratch1
│   └── phoneme-recognition     Another clone of this repository.
│       │                       This is where the data lives (copied with rsync).
│       ├── data
│       └── ...
├── scratch2
└── scratch3
```

Our usage of storage looks something like this as of writing:

```
Your HOME quota:
    Path                    Size  Used  Avail  Use%
    /home/lorenz.marqueses  100G  1.6M  100G   1%

Your SCRATCH quotas:
    Path                        used    quota  limit  grace  files  quota  limit  grace
    /scratch1/lorenz.marqueses  12.27G  2.5T   3T     -      5757   0      0      -
    /scratch2/lorenz.marqueses  12.27G  2.5T   3T     -      5757   0      0      -
    /scratch3/lorenz.marqueses  4k      500G   1T     -      1      0      0      -
     Note: scratch1 and scratch2 are in the same storage device.
```

> [!CAUTION]
> HPC specifies **not** to run jobs in `/home`, so use the clone in `/scratch1`.

In case something goes wrong, our current setup on HPC can be reproduced as
follows:

### Cloning this repository into `/scratch`

```bash
cd ~/scratch1
git clone https://github.com/dlsu-cse-nlp/phoneme-recognition.git
```

### Uploading datasets

Run the following at the repository root to copy data to HPC with `rsync`
(ensure you create the `models/g2p/` directory on HPC first):

```bash
D=[COARE USERNAME]:/scratch1/lorenz.marqueses/phoneme-recognition
rsync -avhP data/fleurs/audio/  $D/data/fleurs/audio/
rsync -avhP data/fsc/parquet/   $D/data/fsc/parquet/
rsync -avhP data/nexdata/audio/ $D/data/nexdata/audio/
rsync -avhP models/g2p/         $D/models/g2p/
```

### Installing `uv` on an HPC compute node

COARE provides a module for `conda`, but since I prefer using `uv` and they
permit installing tools anyway, see below for how to install it on HPC:

```bash
salloc --partition=debug --qos=debug_default --nodes=1 --ntasks=4 --time=01:00:00
srun --pty /bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/scratch1/lorenz.marqueses/uv_cache
export HF_HOME=/scratch1/lorenz.marqueses/hf_cache
```

From here, you can now `uv sync` in the repository:

```bash
cd ~/scratch1/phoneme-recognition
uv sync --frozen
```

Call `exit` twice to leave the `srun` shell and to release the `salloc`.

> [!NOTE]
> You will need to run the environmment variable `export`s every time you login
> to a new interactive session.
