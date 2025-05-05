REM Adding prepocessing stage
dvc stage add -n preprocess ^
    -p preprocess.input ^
    -p preprocess.output.train ^
    -p preprocess.output.test ^
    -d ./src/preprocessing.py ^
    -d ./data/raw/tourism.csv ^
    -o ./data/preprocessed/train.csv ^
    -o ./data/preprocessed/test.csv ^
    python ./src/preprocessing.py

REM Adding training stage
dvc stage add -n train ^
    -p train.data ^
    -p train.models.base_model ^
    -p train.models.tuned_model ^
    -p train.random_state ^
    -d ./src/training.py ^
    -d ./data/preprocessed/train.csv ^
    -o models/base_rf.pkl ^
    -o models/tuned_rf.pkl ^
    python ./src/training.py

REM Adding Evaluation stage
dvc stage add -n evaluate ^
    -d ./src/evaluate.py ^
    -d models/base_rf.pkl ^
    -d models/tuned_rf.pkl ^
    -d ./data/preprocessed/test.csv ^
    python ./src/evaluate.py

REM After running this and the dvcyaml is created, run 'dvc repro' to actually trigger the pipeline.

