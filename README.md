# cnn-eyetrack-colab
CNN eyetracker implemented in Jupyter / Colab

# About
I've always wanted to create an eyetracker which can follow my eyes in a very general direction - in order to focus windows on the computer screen automatically. This would allow working with multiple screens and windows without taking your fingers off your keyboard.

This is the very first working concept. The goal was to create a tracker based on a CNN which can predict the screen coordinates a user is focusing as accurately as possible.

## Architecture
This first implementation uses [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/) as a feature extractor. The resulting sub-regions are then fed into another CNN, together with some other features:

```
(96x48 image)       (image-coordinates of both eyes)
     |                              |
4 Conv layers                2 dense layers
     |                              |
     +---------- concat ------------+
                   |
            4 dense layers
                   |
   (x/y screen-coordinate regression)
```

## Performance
Currently the best performance was achieved by a simple SGD (lr=0.01, decay=1e-6, momentum=0.9) trained for around 1200 epochs.

For prediction accuracy, the numbers below should give a hint (see section "Eval" in notebook for pics):

```
The best 10:
#662: squared error 0.00001 -> pred (1834px, 420px) | actual (1820px, 420px)
#50: squared error 0.00002 -> pred (911px, 325px) | actual (920px, 320px)
#551: squared error 0.00004 -> pred (3449px, 814px) | actual (3440px, 820px)
#269: squared error 0.00005 -> pred (1897px, 524px) | actual (1920px, 520px)
#741: squared error 0.00005 -> pred (2263px, 115px) | actual (2240px, 120px)
#348: squared error 0.00005 -> pred (2312px, 719px) | actual (2340px, 720px)
#244: squared error 0.00005 -> pred (1292px, 120px) | actual (1320px, 120px)
#425: squared error 0.00005 -> pred (418px, 912px) | actual (420px, 920px)
#443: squared error 0.00007 -> pred (2242px, 411px) | actual (2240px, 420px)
#491: squared error 0.00008 -> pred (1126px, 129px) | actual (1120px, 120px)

The mediocre 10:
#343: squared error 0.00551 -> pred (1059px, 98px) | actual (1120px, 20px)
#672: squared error 0.00552 -> pred (1643px, 143px) | actual (1720px, 220px)
#328: squared error 0.00552 -> pred (756px, 651px) | actual (1020px, 620px)
#127: squared error 0.00553 -> pred (406px, 494px) | actual (520px, 420px)
#799: squared error 0.00554 -> pred (1382px, 790px) | actual (1520px, 720px)
#545: squared error 0.00556 -> pred (406px, 516px) | actual (120px, 520px)
#754: squared error 0.00563 -> pred (3544px, 861px) | actual (3740px, 920px)
#219: squared error 0.00564 -> pred (3227px, 611px) | actual (2940px, 620px)
#39: squared error 0.00565 -> pred (2061px, 391px) | actual (1920px, 320px)
#648: squared error 0.00569 -> pred (2359px, 798px) | actual (2440px, 720px)

The worst 10:
#204: squared error 0.03327 -> pred (406px, 484px) | actual (20px, 320px)
#879: squared error 0.03452 -> pred (3326px, 157px) | actual (3740px, 320px)
#281: squared error 0.03481 -> pred (403px, 390px) | actual (20px, 220px)
#38: squared error 0.03530 -> pred (409px, 691px) | actual (20px, 520px)
#812: squared error 0.05104 -> pred (3462px, 793px) | actual (3140px, 1020px)
#703: squared error 0.05410 -> pred (1500px, 769px) | actual (1620px, 520px)
#357: squared error 0.05871 -> pred (407px, 571px) | actual (120px, 820px)
#9: squared error 0.08608 -> pred (1104px, 238px) | actual (1920px, 20px)
#692: squared error 0.09748 -> pred (2719px, 753px) | actual (2540px, 420px)
#52: squared error 0.11113 -> pred (1619px, 780px) | actual (1620px, 420px)
```
"The worst 10" include mostly very bad angles or completely failed detections by MTCNN.

**TLDR: most detections have a very reasonable accuracy for the task which is to accurately predict the window a user is focussing.**

# Running
## Dataset
To recreate the tracker you need to generate a dataset first.
As an example, here is the dataset which I used (an archive of approx. 5GB and a .json file):

https://drive.google.com/file/d/1-OGHcz8oKsFqHm8KQISk9V_5qVqv9nYT/view?usp=sharing
https://drive.google.com/file/d/1ppstumDgSOqzXBNP04DnJH41uhOp6CuB/view?usp=sharing

Place them in a folder structure as below!

This dataset has been created with the capture tool to be found in https://github.com/r-or/cnn-eyetrack.

## Folder structure 
The folder structure is the following:
```
My Drive/ (google drive root)
 |--4proj/
    |--eyetrack/
       |--models/<dataset-name>/
       |  |--logs/ (tensorboard-logs are placed in here)
       |  |   |--run1/
       |  |   | ...
       |  |   +--runN/
       |  +--chkp/ (checkpoints are placed in here)
       |      |--run1/
       |      | ...
       |      +--runN/
       +--source_data/
          |--raw/
          |  |--<dataset-name>.json
          |  +--<dataset-name>.zip
          +--<dataset-name>.npy (this is auto-generated to cache the raw files)
```
This can of course be modified by editing the appropriate variables in the section "Google Drive access and paths".

## Running detection
Once set up like above everything should run correctly. For the first run (or if the flag REFRESH_DATA is set) face extraction has to be performed and will create a .npy file with the appropriate features extracted. This takes a while. Any subsequent runs will be quicker. The .npy will be uploaded to your google Drive and recovered, so if the training set changes REFRESH_DATA must be set to overwrite it.

Within the "Training" section multiple params can be modified and batch training is possible. Each run will be placed in a different folder inside of eyetrack/models/. All model data will be uploaded to your google Drive.
