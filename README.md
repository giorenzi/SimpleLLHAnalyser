# SimpleLLHAnalyser

## Prerequisites:
numpy, scipy, pyminuit

## What you need:
- Binned PDFs for signal and background in any shape; in units of events per second (rate); note: this typically encodes a normalisation of the signal expectation, i.e. a cross section or annihilation rate
- The livetime of the analysis
- If you want to use the effective likelihood from [arXiv:1901.04645], you also need the signal and background PDFs with squared weights

## What you get:
- (For now only) Sensitivities:

The function `CalculateSensitivity(nTrials, CL)` returns a dictionary with sensitivities at the chosen confidence level. This dictionary contains a list of all TS values `'TS_dist'`, as well the `'median'` sensitivity as well as the 1 and 2 sigma bands `'error_68_low','error_68_high','error_95_low','error_95_high'`. The sensitivity will be provided in the same normalisation that the signal pdf was defined in.

## Example:
```python
import LLHAnalyser
analysis = LLHAnalyser.Profile_Analyser()
analysis.setLivetime(livetime)
analysis.loadBackgroundPDF(bkgpdf)
analysis.loadSignalPDF(sigpdf)
analysis.setLLHtype('Poisson')
conf_level = 90
sens = analysis.CalculateSensitivity(100000, conf_level)
analysis.loadUncertaintyPDFs(bkgpdf_weight2,sigpdf_weight2)
analysis.setLLHtype('Effective')
sens = analysis.CalculateSensitivity(100000, conf_level)
```
