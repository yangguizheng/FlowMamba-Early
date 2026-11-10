# Screened variants with validation and test scores

Seed 20260921. Validation scores are those of the selected checkpoint, read from the training logs (four decimals); test scores are from the summary CSVs of the training scripts.

| Model | Dataset | Exit | Validation macro-F1 | Test macro-F1 | Selected epoch | Tied epochs |
|---|---|---:|---:|---:|---:|---:|
| FlowMamba-Early, as selected | USTC-TFC2016 | 1 | 0.8893 | 0.8801 | 17 | 1 |
| FlowMamba-Early, as selected | USTC-TFC2016 | 2 | 0.8762 | 0.8671 | 17 | 1 |
| FlowMamba-Early, as selected | USTC-TFC2016 | 3 | 0.9490 | 0.9470 | 17 | 1 |
| FlowMamba-Early, as selected | USTC-TFC2016 | 5 | 0.9406 | 0.9293 | 17 | 1 |
| FlowMamba-Early, as selected | CipherSpectrum | 3 | 0.6596 | 0.6600 | 7 | 1 |
| FlowMamba-Early, as selected | CipherSpectrum | 5 | 0.8353 | 0.8339 | 7 | 1 |
| Exit-specific projections | USTC-TFC2016 | 1 | 0.8894 | 0.8864 | 9 | 1 |
| Exit-specific projections | USTC-TFC2016 | 2 | 0.8584 | 0.8522 | 9 | 1 |
| Exit-specific projections | USTC-TFC2016 | 3 | 0.9532 | 0.9492 | 9 | 1 |
| Exit-specific projections | USTC-TFC2016 | 5 | 0.9448 | 0.9415 | 9 | 1 |
| Exit-specific projections | CipherSpectrum | 3 | 0.6186 | 0.6252 | 20 | 1 |
| Exit-specific projections | CipherSpectrum | 5 | 0.8227 | 0.8236 | 20 | 1 |
| Learned exit weights | USTC-TFC2016 | 1 | 0.8884 | 0.8881 | 15 | 1 |
| Learned exit weights | USTC-TFC2016 | 2 | 0.8710 | 0.8733 | 15 | 1 |
| Learned exit weights | USTC-TFC2016 | 3 | 0.9536 | 0.9530 | 15 | 1 |
| Learned exit weights | USTC-TFC2016 | 5 | 0.9525 | 0.9393 | 15 | 1 |
| Learned exit weights | CipherSpectrum | 3 | 0.6016 | 0.5972 | 16 | 1 |
| Learned exit weights | CipherSpectrum | 5 | 0.8096 | 0.8123 | 16 | 1 |
| Both changes | USTC-TFC2016 | 1 | 0.8920 | 0.8870 | 11 | 1 |
| Both changes | USTC-TFC2016 | 2 | 0.8724 | 0.8624 | 11 | 1 |
| Both changes | USTC-TFC2016 | 3 | 0.9570 | 0.9436 | 11 | 1 |
| Both changes | USTC-TFC2016 | 5 | 0.9426 | 0.9264 | 11 | 1 |
| Both changes | CipherSpectrum | 3 | 0.6634 | 0.6557 | 17 | 1 |
| Both changes | CipherSpectrum | 5 | 0.8286 | 0.8288 | 17 | 1 |
| Two shared and two exit-specific layers | USTC-TFC2016 | 1 | 0.8857 | 0.8853 | 5 | 1 |
| Two shared and two exit-specific layers | USTC-TFC2016 | 2 | 0.8662 | 0.8610 | 5 | 1 |
| Two shared and two exit-specific layers | USTC-TFC2016 | 3 | 0.9398 | 0.9381 | 5 | 1 |
| Two shared and two exit-specific layers | USTC-TFC2016 | 5 | 0.9305 | 0.8495 | 5 | 1 |
| Two shared and two exit-specific layers | CipherSpectrum | 3 | 0.6795 | 0.6780 | 6 | 1 |
| Two shared and two exit-specific layers | CipherSpectrum | 5 | 0.8535 | 0.8449 | 6 | 1 |

Validation macro-F1 by model (rows) and dataset-exit (columns):

```
dataset                                 CipherSpectrum         USTC-TFC2016                        
exit                                                 3       5            1       2       3       5
model                                                                                              
Both changes                                    0.6634  0.8286       0.8920  0.8724  0.9570  0.9426
Exit-specific projections                       0.6186  0.8227       0.8894  0.8584  0.9532  0.9448
FlowMamba-Early, as selected                    0.6596  0.8353       0.8893  0.8762  0.9490  0.9406
Learned exit weights                            0.6016  0.8096       0.8884  0.8710  0.9536  0.9525
Two shared and two exit-specific layers         0.6795  0.8535       0.8857  0.8662  0.9398  0.9305
```
