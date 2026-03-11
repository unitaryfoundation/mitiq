# Twirled Readout Error eXtinction

Twirled Readout Error eXtinction (TREX) is a readout error mitigation
technique that applies random X gates before measurement (readout
twirling) and uses calibration data to correct the resulting expectation
values. Unlike [REM](rem.md), TREX does not require an explicit inverse
confusion matrix. Instead, it estimates readout error correction factors
directly from calibration circuits. In this sense, TREX is "model-free":
it does not need a pre-characterized noise model (e.g., a confusion
matrix) as input. The calibration circuits are used to *estimate*
correction factors on-the-fly, not to build a reusable noise model.

For more discussion of the theory of TREX, see the section
[What is the theory behind TREX?](trex-5-theory.md).

Below you can find sections of the documentation that address the
following questions:

```{toctree}
---
maxdepth: 1
---
trex-1-intro.md
trex-2-use-case.md
trex-3-options.md
trex-4-low-level.md
trex-5-theory.md
```
