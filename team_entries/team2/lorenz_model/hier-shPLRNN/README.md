# Learning Interpretable Hierarchical Dynamical Systems Models From Time Series Data [[ICLR 2025]](https://openreview.net/forum?id=Vp2OAxMs2s)

## Requirements

We include the `requirements.txt` file to clone our python environment. Simply run
```
pip install -r requirements.txt
```

## Usage

### Training
To start training a model, use the `main.py` file. Any hyper-parameters can be passed as command line arguments. Refer to the file, to see which hyper-parameters exist.
```
python main.py
```

Running multiple trainings, potentially in parallel, can be done via
```
python ubermain.py
```
any hyper-parameters must then be supplied in list format, see the file for more information. This also allows to do simple hyper-parameter grid searches.

### Evaluation
Trained models can be evaluated by
```
python main_eval.py --model_path <MODEL_PATH> --save_path <SAVE_PATH>
```
which will load all trained models in the `<MODEL_PATH>` directory and evaluate them in terms of state space divergence and average hellinger distance. The results will be saved to a file in `<SAVE_PATH>`.

## Citation

If you find the repository and/or paper helpful for your own research, please cite our work.
```
@inproceedings{
    brenner2025learning,
    title={Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data},
    author={Manuel Brenner and Elias Weber and Georgia Koppe and Daniel Durstewitz},
    booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://openreview.net/forum?id=Vp2OAxMs2s}
}
```

## Acknowledgements

This work was funded by the European Union’s Horizon 2020 programme under grant agreement 945263 (IMMERSE), by the German Ministry for Education \& Research (BMBF) within the FEDORA (01EQ2403F) consortium, by the Federal Ministry of Science, Education, and Culture (MWK) of the state of Baden-Württemberg within the AI Health Innovation Cluster Initiative and living lab (grant number 31-7547.223-7/3/2), by the German Research Foundation (DFG) within the collaborative research center TRR-265 (project A06  \& B08) and by the Hector-II foundation.