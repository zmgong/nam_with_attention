# Neural Additive Models for InterpretML

  **[Overview](#overview)**
| **[Paper](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf)**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/nam)
![GitHub license](https://img.shields.io/github/license/lemeln/nam)

*This repo is forked from AmrMKayid/nam to be adapted for integration of NAMs and Multitask NAMs into InterpretML.*

Neural Additive Models (NAMs) are an interpretable ("glassbox") machine learning technique that jointly learns a separate network on each input feature.

Please see https://github.com/google-research/google-research/tree/master/neural_additive_models for the original TensorFlow implementation of single task NAMs used in the paper.

## Example Usage

```python
from nam.wrapper import NAMClassifier

model = NAMClassifier(
            num_epochs=1000,
            num_learners=20,
            metric='auroc',
            early_stop_mode='max',
            monitor_loss=False,
            n_jobs=10,
            random_state=random_state
        )

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)
sk_metrics.roc_auc_score(y_test, pred)
```
See '''classification.ipynb''' and '''regression.ipynb''' for more detail.

## Acknowledgements

This repo, which was created by Levi Melnick, forks AmrMKayisd/nam, which is maintained by Amr Kayid and Nicholas Frosst. Much of the code has been significantly rewritten to be included in InterpretML, but Kayid and Frosst's work provided an excellent starting point.

```bibtex
@article{agarwal2020neural,
  title={Neural additive models: Interpretable machine learning with neural nets},
  author={Agarwal, Rishabh and Melnick, Levi and Frosst, Nicholas and Zhang, Xuezhou and Caruana, Rich and Hinton, Geoffrey E},
  journal={arXiv preprint arXiv:2004.13912},
  year={2021}
}
```
