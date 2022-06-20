# boostcamp_pstage10

# Docker
```bash
docker run -it --gpus all --ipc=host -v $PWD:/opt/ml/code -v ${dataset}:/opt/ml/data placidus36/pstage4_lightweight:v0.1 /bin/bash
```

# Run
## 1. train
python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config}

## 2. inference(submission.csv)
python inference.py --model_config /home/kyungmin/pstage4/code/exp/2021-06-07_10-19-01/model.yml --weight /home/kyungmin/pstage4/code/exp/2021-06-07_10-19-01/best.pt --img_root /home/kyungmin/pstage4/data/test --data_config /home/kyungmin/pstage4/code/configs/data/taco.yaml

# optuna table
optuna dashboard --storage postgresql://optuna:lkm961296@101.101.210.70:6013/optuna --study-name pstage_automl1