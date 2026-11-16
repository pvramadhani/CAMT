# Camera-Aware Mutual Teaching (CAMT)

**Camera-Aware Mutual Teaching (CAMT)** is a research implementation for **unsupervised domain adaptation (UDA) in person re-identification (Re-ID)**. This project combines mutual teacher–student learning with camera-aware proxy associations. The target-domain identities are **not used as training labels**; target camera identifiers are used to construct camera-aware proxies.

## 1. Overview

MMT uses two student networks and their exponential moving average (EMA) teachers to refine cluster-based pseudo labels. CAMT retains this mutual-teaching framework and introduces a **shared camera-aware proxy memory** and an additional proxy-based contrastive objective.

CAMT follows this training cycle:

1. **Source-domain pretraining:** Independently pretrain two ResNet-50 models with labeled source data.
2. **Target pseudo labeling:** Extract target features with both EMA teachers; combine the teacher features and use k-reciprocal/Jaccard distance with DBSCAN to construct target pseudo identities.
3. **Camera-aware proxy construction:** Split each pseudo-identity cluster by target-camera ID. Initialize one shared proxy memory from aggregated EMA-teacher features.
4. **Joint training:** Update the two students using MMT hard/soft supervision and camera-aware proxy contrastive supervision.
5. **Teacher and memory updates:** Update each EMA teacher from its corresponding student; update the shared proxy memory after the student optimization step.
6. **Iterative refinement:** Regenerate pseudo labels and camera-aware proxies from updated teacher representations at the next epoch.

### Camera-aware association

- **Offline association:** Positives are camera-aware proxies that share the anchor's DBSCAN pseudo identity.
- **Online association:** Feature-based cross-camera proxy associations provide additional positives (up to three in the documented configuration), helping mitigate delayed or imperfect clustering assignments.
- **Negative association:** Competing proxies provide contrastive negatives.
- **Shared memory:** Both student networks use the same camera-aware proxy memory; the memory is not a separate inference-time network.

These components supplement rather than replace MMT's original hard and soft losses. Online associations influence representation learning; they do not directly overwrite the current epoch's DBSCAN labels.

### Training objective

CAMT adds a camera-aware proxy term to the original MMT objective:

~~~text
L_CAMT = L_MMT + beta * (L_proxy,student1 + L_proxy,student2)

L_proxy = 0.5 * (L_offline + L_online)
~~~

Here, beta is the proxy-loss weight. The MMT component retains hard pseudo-label classification/triplet learning and peer-teacher soft classification/triplet learning.

## 2. Experimental configuration

The following settings document the current ResNet-50 / DBSCAN experiment. Hardware, dependency versions, and any deviations should be recorded with each released run.

| Component | Setting |
| --- | --- |
| Backbone | ResNet-50 with BN-neck |
| Framework | Two students + two EMA teachers |
| Source pretraining | Two independent random seeds |
| Target clustering | k-reciprocal/Jaccard distance + DBSCAN |
| Target training | 40 epochs |
| Iterations per epoch | 400 |
| Global batch size | 32 |
| Instances per pseudo identity | 4 |
| Student optimizer | Adam |
| Learning rate | 0.00035 |
| Weight decay | 0.0005 |
| EMA coefficient | 0.999 |
| Soft CE weight | 0.5 |
| Soft triplet weight | 0.8 |
| Proxy-loss weight (beta) | 0.1 |
| Proxy temperature | 0.05 |
| Proxy-memory momentum | 0.2 |
| Negative proxies | 50 (documented method setting; not a CLI flag) |
| Online cross-camera positives | Up to 3 (documented method setting; not a CLI flag) |
| Proxy association | Equal offline/online contributions |

**Important:** The exact training script, package versions, DBSCAN settings, and checkpoint serialization should be checked against the released code before claiming a fully reproduced result. Some values above are method settings rather than individually exposed command-line arguments.

## 3. Installation

### Requirements

- Linux environment with CUDA-compatible NVIDIA GPU
- Conda or equivalent Python environment manager
- Python 3.8 (used in the local development environment)
- PyTorch, torchvision, FAISS, and the dependencies required by the released source code

Exact PyTorch, torchvision, CUDA, and FAISS versions will be added with the environment specification. The old Python 3.6 / PyTorch 1.1 badges from the original MMT README are **not** claims about this CAMT environment.

Clone the repository:

~~~bash
git clone https://github.com/pvramadhani/CAMT.git
cd CAMT
~~~

Create an environment:

~~~bash
conda create -n camt python=3.8 -y
conda activate camt
~~~

**After the code and dependency files have been uploaded**, install the included package (if the MMT-derived setup.py is retained):

~~~bash
python -m pip install -e .
~~~

On systems where user-site packages may conflict with the Conda environment, use:

~~~bash
export PYTHONNOUSERSITE=1
~~~

## 4. Prepare datasets

CAMT follows the MMT UDA data layout for [Market-1501](https://openaccess.thecvf.com/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), and [MSMT17](https://arxiv.org/abs/1711.08565).

Download these datasets from their authorized distribution channels and organize them as follows:

~~~text
CAMT/
└── examples/
    └── data/
        ├── dukemtmc/
        │   └── DukeMTMC-reID/
        ├── market1501/
        │   └── Market-1501-v15.09.15/
        └── msmt17/
            └── MSMT17_V1/
~~~

Dataset archive names and MSMT17 versions must match those expected by the released dataset loader. Dataset images and identity annotations must not be committed to this repository. Source identity labels may be used for source pretraining; target identity labels must not be used during adaptation.

## 5. Training

> **Code-release prerequisite:** The commands in this section refer to the locally used or MMT-derived files examples/source_pretrain.py and examples/mmt_train_dbscan_cap.py. They cannot run from the current README-only repository. Publish and check these files and their dependencies first.

### Stage I — Source-domain pretraining

Train **two independently initialized models** on the labeled source domain. For DukeMTMC-reID to Market-1501:

~~~bash
for seed in 1 2; do
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python examples/source_pretrain.py \
    -ds dukemtmc -dt market1501 -a resnet50 \
    --seed $seed --margin 0.0 \
    --num-instances 4 -b 32 -j 4 \
    --warmup-step 10 --lr 0.00035 \
    --milestones 40 70 \
    --iters 100 --epochs 80 --eval-step 40 \
    --logs-dir logs/dukemtmcTOmarket1501/resnet50-pretrain-$seed
done
~~~

Expected source checkpoint paths:

~~~text
logs/dukemtmcTOmarket1501/
├── resnet50-pretrain-1/model_best.pth.tar
└── resnet50-pretrain-2/model_best.pth.tar
~~~

The two initialization paths must correspond to the selected **source–target experiment**, and must not be silently reused from an unrelated experiment.

### Stage II — CAMT adaptation: Duke to Market

Using the CAMT local entrypoint, the documented B32 configuration is:

~~~bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python examples/mmt_train_dbscan_cap.py \
  -ds dukemtmc -dt market1501 -a resnet50 \
  --num-instances 4 -b 32 -j 4 \
  --lr 0.00035 --iters 400 --epochs 40 \
  --soft-ce-weight 0.5 --soft-tri-weight 0.8 \
  --cap-weight 0.1 --cap-temp 0.05 --cap-momentum 0.2 \
  --init-1 logs/dukemtmcTOmarket1501/resnet50-pretrain-1/model_best.pth.tar \
  --init-2 logs/dukemtmcTOmarket1501/resnet50-pretrain-2/model_best.pth.tar \
  --logs-dir logs/dukemtmcTOmarket1501/resnet50-CAMT-B32
~~~

The CAP flags shown above were used in the local implementation. Additional settings, including negative-proxy selection and online association, should be documented from the actual implementation rather than represented by invented command-line flags.

### Example — Duke to MSMT17

First pretrain the two source models for the Duke-to-MSMT17 experiment by changing the target to msmt17 and using the matching output directory:

~~~bash
for seed in 1 2; do
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python examples/source_pretrain.py \
    -ds dukemtmc -dt msmt17 -a resnet50 \
    --seed $seed --margin 0.0 \
    --num-instances 4 -b 32 -j 4 \
    --warmup-step 10 --lr 0.00035 \
    --milestones 40 70 \
    --iters 100 --epochs 80 --eval-step 40 \
    --logs-dir logs/dukemtmcTOmsmt17/resnet50-pretrain-$seed
done
~~~

Then launch CAMT:

~~~bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python examples/mmt_train_dbscan_cap.py \
  -ds dukemtmc -dt msmt17 -a resnet50 \
  --num-instances 4 -b 32 -j 4 \
  --lr 0.00035 --iters 400 --epochs 40 \
  --soft-ce-weight 0.5 --soft-tri-weight 0.8 \
  --cap-weight 0.1 --cap-temp 0.05 --cap-momentum 0.2 \
  --init-1 logs/dukemtmcTOmsmt17/resnet50-pretrain-1/model_best.pth.tar \
  --init-2 logs/dukemtmcTOmsmt17/resnet50-pretrain-2/model_best.pth.tar \
  --logs-dir logs/dukemtmcTOmsmt17/resnet50-CAMT-B32
~~~

**Checkpoint warning:** For Duke-to-MSMT17, the initialization paths must point to dukemtmcTOmsmt17 source-pretrained models, not market1501TOmsmt17. In the original MMT README, the Duke-to-MSMT17 DBSCAN example also accidentally specified market1501 as its target; this has been corrected here.

### Other source–target directions

To run another direction (for example, Market-to-Duke or Market-to-MSMT17), consistently change:

- The -ds and -dt dataset arguments.
- Both source-pretrained checkpoint paths.
- The output logs directory.
- The dataset loader configuration, if a dataset version differs.

Do **not** change only the dataset name while leaving unrelated source-pretrained model paths.

## 6. Evaluation

CAMT's documented evaluation uses the **first EMA teacher** for inference, with L2-normalized BN-neck features and standard person Re-ID evaluation (mAP, Rank-1, Rank-5, and Rank-10). DBSCAN and the proxy memory are not needed at inference time.

The upstream MMT test interface is:

~~~bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python examples/test_model.py \
  -b 256 -j 8 \
  --dataset-target market1501 \
  -a resnet50 \
  --resume /path/to/teacher1_final_epoch_checkpoint.pth.tar
~~~

**This is an interface example, not a verified CAMT evaluation command.** The released CAMT implementation must ensure that the file provided to --resume contains the first EMA teacher in the format expected by the evaluator. A generic student checkpoint or MMT model_best.pth.tar must **not** be assumed to be the final-epoch EMA teacher.

For reproducible reporting:

- Use the **same evaluation protocol** for CAMT and all MMT baselines.
- Do not use target test identities during adaptation.
- Do not select the best checkpoint or tune hyperparameters using target test-set mAP.
- Report the evaluation checkpoint, dataset split, seed, preprocessing, and whether re-ranking was applied.
- The documented CAMT protocol uses the **final epoch** and **no inference-time re-ranking**.
