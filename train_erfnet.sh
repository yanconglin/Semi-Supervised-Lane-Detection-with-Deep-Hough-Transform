python3 -u train_erfnet.py CULane ERFNet train_gt val_gt \
                        --lr 0.01 \
                        --gpus 0 \
                        --resume /tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/lanes/ERFNet-CULane-HTIHT-scratch/5513873/trained/_erfnet_model_best.pth.tar \
                        -j 4 \
                        -b 14 \
                        --epochs 12 \
                        --eval-freq 1 \
                        --img_height 208 \
                        --img_width 976 \
2>&1|tee train_erfnet_culane_ht.log
