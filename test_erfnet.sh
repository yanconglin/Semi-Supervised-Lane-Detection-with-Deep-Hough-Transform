python3 -u test_erfnet2.py CULane ERFNet test_img test_img \
                          --gpus 0 \
                          --resume trained/_erfnet_model_best.pth.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -b 28 \
                          -j 4 \
2>&1|tee test_erfnet2.log

