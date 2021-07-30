## Source code for [Semi-Supervised Lane Detection with Deep Hough Transform](https://arxiv.org/abs/2106.05094), ICIP2021.
This repository is built on top of [ERFNet-CULane-PyTorch](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch). Many thanks to the authors for sharing the code. 
### Requirements
- [PyTorch 1.3.0](https://pytorch.org/get-started/previous-versions/).
- Matlab (for tools/prob2lines), version R2014a or later.
- Opencv (for tools/lane_evaluation), version 2.4.8 (later 2.4.x should also work).

### Before Start

1. Please follow [ERFNet-CULane-PyTorch](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch) to prepare the CULane dataset.
2. Please set the dataset path in the [dataloader](https://github.com/yanconglin/Semi-Supervised-Lane-Detection-with-Deep-Hough-Transform/blob/a5562e0c308a7c0e30d360bc380bdfc3309b032d/dataset/voc_aug.py#L10). 

### Testing
The model in repo is the ERFNet-HT+L_{HT}.
1. Download the well-trained model (on 1% labeled+99% unlabeled set) [semi_erfnet_model_best_1_99.pth.tar](https://surfdrive.surf.nl/files/index.php/s/r45otmIzGTQkWLu) to `./trained`
    ```Shell
    cd $ERFNet_ROOT/trained
    ```
   The trained model has already been there.

2. Run test script
    ```Shell
    cd $ERFNet_ROOT
    sh ./test_erfnet.sh
    ```
    Testing results (probability map of lane markings) are saved in `experiments/predicts/` by default.

3. Get curve line from probability map
    ```Shell
    cd tools/prob2lines
    matlab -nodisplay -r "main;exit"  # or you may simply run main.m from matlab interface
    ```
    The generated line coordinates would be saved in `tools/prob2lines/output/` by default.

4. Calculate precision, recall, and F-measure
    ```Shell
    cd $ERFNet_ROOT/tools/lane_evaluation
    make
    sh Run.sh   # it may take over 30min to evaluate
    ```
    Note: `Run.sh` evaluate each scenario separately while `run.sh` evaluate the whole. You may use `calTotal.m` to calculate overall performance from all senarios.  
    By now, you should be able to reproduce the result.
    
### Training
The model in repo is the ERFNet-HT+L_{HT}.
1. Train the ERFNet_HT on the 1% subset in a fully supervised manner, or you can simply download the checkpoint [supervised_erfnet_model_best_1.pth.tar](https://surfdrive.surf.nl/files/index.php/s/r45otmIzGTQkWLu).
    ```Shell
    cd $ERFNet_ROOT/pretrained
    ```
2. Load the checkpoint from step 1, and train the ERFNet_HT model on the full dataset (1% labeled + 99% unlabeled), in a semi-supervised manner.
    ```Shell
    cd $ERFNet_ROOT
    sh ./train_erfnet.sh
    ```
    The training process should start and trained models would be saved in `trained` by default.  
    Then you can test the trained model following the Testing steps above. If your model position or name is changed, remember to set them to yours accordingly.

### Cite Deep Hough-Transform Line Priors

If you find our paper useful in your research, please consider citing:
```bash
@article{lin2021semi,
  title={Semi-supervised lane detection with Deep Hough Transform},
  author={Lin, Yancong and Pintea, Silvia L and van Gemert, Jan C},
  booktitle={ICIP 2021},
  year={2021}
}
```
