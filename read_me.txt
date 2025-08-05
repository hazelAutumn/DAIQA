Three version of model:
- MyModel_v1_2: output two type of labels: quality and distortion
- MyModel_v1_0: outputs only distortion prediction
- MyModel_v1_4: output only quality score


For training the model, please take a look at bash files under directory submit_jobs, 
in which xxx10xx.sh is for training MyModel_v1_0, xxx14xxx.sh is for training MyModel_v1_4.
The source code of each model are in src/model/xxx.py

in which xxx10xx.sh is for training MyModel_v1_0, xxx14xxx.sh is for training MyModel_v1_4.
The pretrained model for MyModel_v1_0 and one pretrained model for are saved in ckpt/



In this project, we use Wandb for logging the results.