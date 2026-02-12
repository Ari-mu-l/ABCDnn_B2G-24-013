# ABCDnn_B2G-24-013

Note: Scripts used for studies and checks during the review process are under the `DeprecatedScripts/` directory.

## General Procedures
### Prepare root files for training
- Update sample information `in samples.py`
- Run `format_ntuple.py` to make root files for training

### Train
- Configuration (ML design, control/transform variables, plot settings): `config.py`
- ML code: `abcdnn_mmd.py`
- Train: `train_abcdnn_mmd.py` or `train_abcdnn_hpo_mmd.py` (better for hyperparameter optimization)
- Plot training: `plot_training.py` (full ST), `plot_training_validation.py` (low ST), or `plot_training_countervalidation.py` (high ST)

### 2D Correction
- Make 2D histograms: `make2DHist_abcdnn.py`
- Get alpha ratio for normalization: `getAlphaRatio.py`
- Create corrected histograms and relavent uncertainty histograms: `makeUncert2D_abcdnn_1Dsmooth_dynamicST.py`
  - Produce relavent 1D and 2D plots
  - Produce a new root file with postfix '_modified.root'
- Add ABCDnn histograms to template files: `python3 applyHist2D_abcdnn.py`

### Misc
- Training results saved in `Results/`
- To make 2D plots of the control variables (CR and SR regions),run `draw2DHist.py`


