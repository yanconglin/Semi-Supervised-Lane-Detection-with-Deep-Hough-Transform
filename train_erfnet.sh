python3 -u train_erfnet.py CULane ERFNet train_gt val_gt \
                        --lr 0.01 \
                        --gpus 0 \
                        --resume /supervised_erfnet_model_best_1.pth.tar \
                        -j 4 \
                        -b 14 \
                        --epochs 12 \
                        --eval-freq 1 \
                        --img_height 208 \
                        --img_width 976 \
2>&1|tee train_erfnet_culane_ht.log
