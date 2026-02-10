# Twirled Readout Error eXtinction (TREX)

Twirled Readout Error eXtinction (TREX) is a model-free readout error
mitigation technique that applies random X gates before measurement
(readout twirling) and uses calibration data to correct the resulting
expectation values. Unlike [REM](rem.md), TREX does not require an
explicit inverse confusion matrix. Instead, it estimates readout error
correction factors directly from calibration circuits.

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
