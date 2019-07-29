# cnn-eyetrack-colab
CNN eyetracker implemented in Jupyter / Colab


## Dataset
To recreate the tracker you need to generate a dataset first.
As an example, here is the dataset which I used (an archive of approx. 5GB and a .json file):
https://drive.google.com/file/d/1-OGHcz8oKsFqHm8KQISk9V_5qVqv9nYT/view?usp=sharing
https://drive.google.com/file/d/1ppstumDgSOqzXBNP04DnJH41uhOp6CuB/view?usp=sharing
Place them in a folder structure as below!

This dataset has been created with the capture tool found in https://github.com/r-or/cnn-eyetrack.

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
