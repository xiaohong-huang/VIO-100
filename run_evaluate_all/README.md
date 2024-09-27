# VIO-100 RUN ALL


## 1. Prerequisites
### 1.1 EVO tool
Installing the evo tool via:
```
pip install evo --upgrade --no-binary evo
```
## 2. Run
### 1.1 Step 1
In run.py, you may need to change the array of "filenames" to be the file names of the ros bags.
### 1.2 Step 2
Run with all datasets:
```
cd ~/catkin_ws/
cp ./src/VIO-100/run_evaluate_all/run.py ./run.py
source devel/setup.bash
python3 run.py
```
## 2. Evaluation
### 1.1 Step 1
In evaluate.py, You may need to change the array of "truth" to be the file names of the ground-truth.
### 1.2 Step 2
Evaluate with all datasets:
```
cd ~/catkin_ws/
cp ./src/VIO-100/run_evaluate_all/evaluate.py ./evaluate.py
python3 evaluate.py

```
