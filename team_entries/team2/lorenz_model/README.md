# Instructions for Lorenz Problem at the ML4DE Hackathon

### 1. Setup
move into the hierarchical shallow piecewise-linear RNN folder
```bash
cd hier-shPLRNN
```

### 2. Data
Augment and prepare the data by using aux functions from ML4DE (see `lorenz_data_generator.py`), the place them into the `data` folder.

### 2. Train
Train the model with the augmented data (see details in `main.py`) with the command below. Should take about 1 hour on a 4080 GPU. But checkpoints are saved saved every 100 steps into the `trained_moels\experiment\<name>\<run>` folder, which can be used for evaluation.

```bash
python main.py --use_gpu --latent_size=10 --train_set_size=5000 --num_workers=24 --seq_len=30 --num_epochs=1000 --name=test --run=1
```

### 2. Eval
Evaluate the model, and save all predictions. Note the model path !
```bash
python main_eval.py --model_path='./trained_models/experiment/test/001'
```

### 3. Submit
Run the following script to prepare predictions for submission tot he leaderboard
```bash
python submit_leaderboard.py
```

