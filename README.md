

## Setup

```
conda create --name multinerf python=3.9
conda activate multinerf

pip install -r requirements.txt
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

pip install tensorstore
sudo apt-get install imagemagick
apt-get install ffmpeg
```
For the use of GPUs or TPUs, you need to visit https://github.com/google/jax for further detail.

## For model 1 training, rendering, and evaluation: 
```
git checkout master

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr

python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr

python -m eval \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```



## For model 2 training, rendering, and evaluation:
```
git checkout model2

python -m train \
  --gin_configs=configs/model_2.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr

python -m render \
  --gin_configs=configs/model_2.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr

python -m eval \
  --gin_configs=configs/model_2.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```


## For model 3 training, rendering, and evaluation:
```
git checkout model3

python -m train \
  --gin_configs=configs/model_3.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr

python -m render \
  --gin_configs=configs/model_3.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr

python -m eval \
  --gin_configs=configs/model_3.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```