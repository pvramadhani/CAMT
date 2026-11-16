export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1

python examples/mmt_train_dbscan.py \
-dt dukemtmc \
-dt market1501 \
-a resnet50 \
-b 32 \
-j 4 \
--num-instances 4 \
--lr 0.00035 \
--iters 400 \
--epochs 40 \
--eval-step 5 \
--soft-ce-weight 0.5 \
--soft-tri-weight 0.8 \
--dropout 0 \
--lambda-value 0 \
--seed 1 \
--init-1 logs/dukemtmcTOmarket1501/resnet50-pretrain-1/model_best.pth.tar \
--init-2 logs/dukemtmcTOmarket1501/resnet50-pretrain-2/model_best.pth.tar \
--logs-dir logs/dukemtmcTOmarket1501/resnet50-MMT-DBSCAN