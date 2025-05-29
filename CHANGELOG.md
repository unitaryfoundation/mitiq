# Changelog

## Version 0.45.1

Fix packaging issue that caused `import mitiq` to fail due to missing VERSION.txt in wheel.

## Version 0.45.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.44.0...v0.45.0))

### Highlights

- Many tutorials are now updated with modern Qiskit code, ready to run on the latest IBM devices thanks to @bdg221!
- A new tutorial demonstrating how to use [UCC](https://github.com/unitaryfoundation/ucc) in conjunction with Mitiq that shows how compilation and mitigation work well together.
- The Virtual Distillation technique now has a complete [user guide](https://mitiq.readthedocs.io/en/stable/guide/vd.html) along with [API-docs](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.vd.vd.execute_with_vd).

#### ✨ Enhancements

- Update get_scale_factors for static scale factors (#2727) [@bdg221]
- Use uv and pyproject.toml for dependencies and configs (#2724) [@bdg221] (this will not impact users of `mitiq`, but it makes development much easier! Come try it out :))

#### 📓 Documentation

- add Virtual Distillation documentation (#2760) [@FarLab]
- add VD to api-doc (#2761) [@natestemen]
- Update tutorials for IBM hardware (#2759) [@bdg221]
- Add tutorial demonstrating basic usage of mitiq & ucc (#2728) [@natestemen]
- Update README.md (#2718) [@lifechange777]
- fix shadows path (#2717) [@natestemen]

#### 📦 Dependency Updates

- Bump pyscf from 2.8.0 to 2.9.0 (#2725) [@dependabot[bot]]


## Version 0.44.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.43.0...v0.44.0))

### Highlights

🚀 This release introduces the first version of the **Virtual Distillation** (VD) technique in Mitiq, which is now available for use!
This technique was prototyped and implemented by a team of students at the University of Amsterdam.
VD uses additional qubits to distill a purer version of the quantum state of interest.
The implementation is in its early stages so lacks support for all `QPROGRAM` types.
Currently only programs written in `cirq` are supported.
We welcome feedback and suggestions for improvement.

```py
from mitiq import vd

vd.execute_with_vd(circuit, execute)
>>> np.array([0.5, 0.5]) # [<Z_0>, <Z_1>] assuming the circuit acts on 2 qubits
```

We've also made further enhancements to the **Layerwise Richardson Extrapolation** (LRE) technique, including support for observables and the `mitiq.Executor` class.
`mitiq.lre` also has two new functions `mitiq.lre.construct_circuits` and `mitiq.lre.combine_results` that allow users to generate circuits and combine results in a more modular way (bringing this module in line with the other error mitigation techniques).
An example workflow for the two step application LRE is shown below:

```py
from mitiq import lre

lre_circuits = lre.construct_circuits(circuit, degree, fold_multiplier)

results = execute(lre_circuits)

lre_result = lre.combine_results(results, circuit, degree, fold_multiplier)
```

### 🚨 Breaking Changes

For uniformity across modules within Mitiq we have renamed the folowing functions:
1. `mitiq.zne.scaled_circuits` -> `mitiq.zne.construct_circuits`
2. `mitiq.ddd.generate_circuits_with_ddd` -> `mitiq.ddd.construct_circuits`
3. `mitiq.pec.generate_sampled_circuits` -> `mitiq.pec.construct_circuits`

With this change, you will find the function `mitiq.<module>.construct_circuits` in all of ZNE, PEC, DDD, LRE, and VD.

We've also bound the version of `numpy` to be less than `2.0.0` due to some conflicts when using this latest major version.
We hope to support `numpy` 2.0.0 in a coming release.

---

If you're interested in error mitigation, check out our upcoming error resilience workshop in NYC!
WERQSHOP: Workshop on Error Resilience Quantum computing (https://werq.shop).

---


#### ✨ Enhancements

- Modularization uniformity for ZNE, PEC, DDD, and LRE (#2709) [@bdg221]
- Add main API entry points for virtual distillation (#2658) [@Jegbrz]
- UFund => UFoundation (#2706) [@natestemen]
- allow list/tuple constructor in executor typehints (#2700) [@natestemen]
- Adding utility functions for Virtual Distillation. (#2698) [@FarLab]
- Auxilliary code for VD to apply a Bi matrix on a circuit (#2650) [@khknopp]
- Adding support for observables in LRE executors (#2681) [@Shivansh20128]
- Adding modularized function generate_circuits_with_ddd (#2618) [@Shivansh20128]
- Allow for Executor class and batched Executors for LRE (#2676) [@bdg221]
- Adding modularized function generate_sampled_circuits (#2619) [@Shivansh20128]
- Default scaling method in modularized ZNE function (#2666) [@purva-thakre]

#### 📓 Documentation

- Adding docstrings to Observable class (#2699) [@Shivansh20128]
- add Virtual Distillation rfc to the contributing page and a new row in the error mitigation techniques in the readme. (#2691) [@FarLab]

#### 📦 Dependency Updates

- bound numpy from above (#2712) [@natestemen]
- Update qiskit-ibm-runtime requirement from ~=0.36.1 to ~=0.37.0 (#2693) [@dependabot]
- Update qiskit-aer requirement from ~=0.15.1 to ~=0.17.0 (#2694) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.90.2 to ~=1.91.0 (#2697) [@dependabot]
- Update qiskit requirement from ~=1.4.1 to ~=1.4.2 (#2696) [@dependabot]
- Bump openfermion from 1.6.1 to 1.7.0 (#2671) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.69.0 to ~=1.90.2 (#2690) [@dependabot]
- Update qibo requirement from ~=0.2.15 to ~=0.2.16 (#2689) [@dependabot]
- Update qiskit-ibm-runtime requirement from ~=0.20.0 to ~=0.36.1 (#2314) [@dependabot]
- Update scipy requirement from <=1.14.1,>=1.10.1 to >=1.10.1,<=1.15.2 (#2673) [@dependabot]
- Update qiskit requirement from ~=1.3.1 to ~=1.4.1 (#2685) [@dependabot]
- Update pyquil requirement from ~=3.5.4 to ~=4.11.0 (#2063) [@dependabot]

## Version 0.43.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.42.0...v0.43.0))

### Highlights

This release marks the first step toward Virtual Distillation (VD) in Mitiq, with an initial helper function that vertically copies a circuit _M_ times.
A team of students at the University of Amsterdam worked for the month of January on implementing the technique that will be integrated into Mitiq over the coming releases.
We also have a new tutorial thanks to @purva-thakre on combining Pauli Twirling and Zero-Noise Extrapolation!

We are currently testing Layerwise Richardson Extrapolation (LRE) on hardware, and work has begun on Probabilistic Error Amplification (PEA)---stay tuned for more updates in future releases!

#### ✨ Enhancements

- First PR for VD, helper function that copies a circuit M times (#2649) [@chrispy-chicken]

#### 📓 Documentation

- added section on modular functions for ZNE (#2657) [@FarLab + @natestemen]
- PT + ZNE tutorial (#2601) [@purva-thakre]
- Changed the badge to Unitary Foundation (#2645) [@muddi900]
- Typos in PEC User Guide (#2637) [@purva-thakre]
- typo corrected from supeconducting to superconducting (#2617) [@vrajan1996]
- Small fixes in docstrings (#2663) [@cosenal]

#### 🧑🏽‍💻 Developer Improvements

- LRE failure for chunking (#2608) [@purva-thakre]
- update version, changelog (#2613) [@purva-thakre]

#### 📦 Dependency Updates

- Bump pyscf from 2.7.0 to 2.8.0 (#2633) [@dependabot]
- Update qibo requirement from ~=0.2.13 to ~=0.2.15 (#2643) [@dependabot]

## Version 0.42.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.41.0...v0.42.0))

### Highlights

🎊 Thanks for a great 2024! Our end of the year recap should be out soon. Subscribe [here](https://forms.gle/wVBki9n9ywA4c1Dj6) to get the newsletter in your mailbox.

🚀 Many thanks to **first time contributors** @gluonhiggs, @JMuff22, @sanketsharma and @Shivansh20128!

- @gluonhiggs added support for some additional Qiskit gates through using gate decomposition for gates that are unavailable
in Cirq. A similar idea was applied to gates not recognized by QASM. 
- @Shivansh20128 added a new page in the documentation for benchmarking circuits.
- @JMuff22 and @sanketsharma corrected typos in the documentation.

💡 A new error-mitigation technique is on its way. Thanks to @Misty-W for the
[Probabilistic Error Amplification RFC](https://docs.google.com/document/d/1l-74EFdMA0CSFUpHjqCyQYb3ZKCmY77seB1_mOZo5Co/edit?usp=sharing)!

🎏 Modular PEC functions are now available, courtesy of @natestemen! These functions allow a
user to generate the intermediary sampled circuits and combine the results in a a two step process. E.g.

```py
from mitiq import pec

circuits = pec.intermediary_sampled_circuits(circuit, representations)

sampled_circuit_results = ...  # execute the circuits on a simulator/hardware/toothbrush/etc

pec_estimate = pec.combine_results(sampled_circuit_results, ...)
```

#### ✨ Enhancements

- Fix converting Rxx and similar Qiskit gates (#2579) [@gluonhiggs]
- Handle unsupported gates (#2585) [@gluonhiggs]
- RFC for Probabilistic Error Amplification technique (#2550) [@Misty-W]
- Address executor and observable incompatibility (#2514) [@bdg221]
- Modularize PEC functionality (#2604) [@natestemen]


#### 🧑🏽‍💻 Developer Improvements

- Fix flaky ZNE factory + observable test (#2602) [@natestemen]
- Ensure further LRE compatibility with non-Cirq circuits (#2599) [@natestemen]
- Throw Erorr on Multiple Measurements per Qubit with Observables (#2593) [@bdg221]

#### 📓 Documentation

- Update Calibration run docstring for API-doc (#2516) [@bdg221]
- Add benchmarking circuits to user guide (#2566) [@Shivansh20128]
- Correct run-on sentences in the API-doc for CDR function docstrings (#2589) [@Shivansh20128]
- Quick fix: Fix typo and headings in classical shadows tutorial (#2574) [@JMuff22]
- Update executor and observable docs (#2594) [@bdg221]
- Add log param to calibration guide (#2568) [@bdg221]
- Fix constant value in ZNE docs (#2591) [@cosenal]
- Fixed typo in documentation (##2611) [@sanketsharma]

#### 📦 Dependency Updates

- Update qiskit requirement from ~=1.2.4 to ~=1.3.1 (#2603) [@dependabot, @cosenal]
- Update qibo requirement from ~=0.2.7 to ~=0.2.13 (#2559) [@dependabot]
- Bump pyqrack from 1.32.21 to 1.32.27 (#2582) [@dependabot, @natestemen]
- Bump codecov/codecov-action from 4 to 5 (#2576) [@dependabot]
- Bump pyqrack from 1.32.11 to 1.32.21 (#2580) [@dependabot]

## Version 0.41.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.40.0...v0.41.0))

### Highlights

📓 The Layerwise Richardson Extrapolation **(LRE) user guide is complete**!
The user guide contains information about both the ins and outs of using the implementation, as well as covering the theory behind the technique so you can make judgements about when to apply the technique.
In addition to finishing the user guide, we also have a new tutorial comparing both the performance and overhead needed for LRE and ZNE.
Big thanks to @purva-thakre and @FarLab for the documentation!

📹 As part of launching LRE we made a **short tutorial video** to showcase the technique, along with how to use it.
Check it out [here](https://www.youtube.com/watch?v=47GWi4h7TWM)!

🧑‍🔬 **First time contributor** @jpacold recreated results from [a paper](https://arxiv.org/abs/2211.08318) on phase transitions in the Ising model.
Both the paper authors, and @jpacold both used Mitiq's ZNE module to apply error mitigation.
This is both an informative tutorial on turning physics problems into something amenable on a quantum computer, and a class in applying error-mitigation.

#### ✨ Enhancements

-  Ensure LRE compatibility with all supported frontends (#2547) [@natestemen]

#### 🧑🏽‍💻 Developer Improvements

- remove failing test; simplify layerwise ZNE tests (#2545) [@natestemen]

### 📞 Call for ideas

We're currently looking into what features we could add to make Mitiq more **noise-aware**.
If you have ideas and features requests in this area, do make a post on the GitHub discussion [here](https://github.com/unitaryfoundation/mitiq/discussions/2193)!

## Version 0.40.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.39.0...v0.40.0))

### Highlights

🔉 A new quantum error-mitigation technique is available in Mitiq!
**Layerwise Richardson Extrapolation** is now available through `mitiq.lre.execute_with_lre`.
Documentation is also available in the user guide, with more advanced docs and demonstrations coming in the next release.
Special thanks to Purva Thakre for this contribution!

🥇 We had two **first time contributions** from @ecarlander and @mbrotos!
Thank you both for your contributions!

🛡️ A **helpful error message** is raised when passing data of the incorrect type to the `MeasurementResult` class, where before it silently gave confusing results.

#### ✨ Enhancements
- LRE Executors (#2499) [@purva-thakre]
- LRE Inference Functions (#2447) [@purva-thakre]
- Raise TypeError when a dictionary is passed to MeasurementResult constructor (#2523) [@natestemen]

#### 📓 Documentation

- Add theory, intro and use case pages of LRE user guide (#2522) [@purva-thakre]
- add QOSS survey banner (#2533) [@natestemen]
- Update broken link in the readme (#2528) [@purva-thakre]
- Move class documentation to class docstrings (#2525) [@natestemen]
- QSE docs cleanup (#2490) [@natestemen]
- Add tags to tutorials (#2467) [@purva-thakre]
- Correct CDR and VNCDR acronyms in example (#2479) [@bdg221]
- added roadmap link to readme (#2468) [@ecarlander]
- Update ibmq-backends.md (#2474) [@mbrotos]

#### 🧑🏽‍💻 Developer Improvements

- Remove `make requirements` (#2481) [@purva-thakre]
- Fix flaky REM test / refactor (#2464) [@natestemen]

#### 📦 Dependency Updates

- Bump pyscf from 2.6.2 to 2.7.0 (#2518) [@dependabot]
- Bump pyqrack from 1.30.24 to 1.30.30 (#2521) [@dependabot]
- Bump pyqrack from 1.30.22 to 1.30.24 (#2497) [@dependabot]
- Update qiskit requirement from ~=1.2.1 to ~=1.2.2 (#2507) [@dependabot]
- Bump qibo from 0.2.10 to 0.2.12 (#2506) [@dependabot]
- Update qiskit-aer requirement from ~=0.15.0 to ~=0.15.1 (#2504) [@dependabot]
- Update qiskit requirement from ~=1.2.0 to ~=1.2.1 (#2503) [@dependabot]
- Bump pyqrack from 1.30.20 to 1.30.22 (#2489) [@dependabot]
- Update qiskit-aer requirement from ~=0.14.2 to ~=0.15.0 (#2484) [@dependabot]
- Bump pyqrack from 1.30.8 to 1.30.20 (#2487) [@dependabot]
- Update cirq-core requirement from <1.4.0,>=1.0.0 to >=1.0.0,<1.5.0 (#2390) [@dependabot]
- Update qiskit requirement from ~=1.1.1 to ~=1.2.0 (#2482) [@dependabot]
- Update scipy requirement from <=1.14.0,>=1.10.1 to >=1.10.1,<=1.14.1 (#2477) [@dependabot]
- Bump pyqrack from 1.30.0 to 1.30.8 (#2476) [@dependabot]
- Bump sphinx from 7.2.6 to 8.0.2 (#2455) [@dependabot]
- Bump qibo from 0.2.9 to 0.2.10 (#2458) [@dependabot]

## Version 0.39.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.38.0...v0.39.0))

### Highlights

We've made updates to our documentation, beginning with the completion of the first section of the Pauli Twirling user guide, which offers a comprehensive introduction to this feature.
Additionally, we've added a new tutorial on CDR (Clifford Data Regression) using [Qrack](https://github.com/unitaryfoundation/qrack/) as an efficient near-Clifford simulator.
This demonstrates a workflow that harnesses the speed of Qrack in the CDR training phase, while providing users with an in-depth look at how to integrate Mitiq and Qrack effectively.

#### 📓 Documentation

- Complete first section of Pauli Twirling user guide (#2454) [@cosenal]
- Hide primary sidebar from certain pages of the documentation (#2424) [@purva-thakre]
- CDR Tutorial with Qrack (#2451) [@bdg221]

#### 🧑🏽‍💻 Developer Improvements

- Separate docs build workflow (#2441) [@purva-thakre]
- clean and git ignore all coverage reports (#2443) [@cosenal]
- Ignore _about.py in pytest coverage (#2379) [@purva-thakre]
- use date to make version unique for testpypi uploads (#2436) [@natestemen]

#### 📦 Dependency Updates

- Update scipy requirement from <=1.13.1,>=1.10.1 to >=1.10.1,<=1.14.0 (#2420) [@dependabot]

## Version 0.38.0 

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.37.0...v0.38.0))

### Highlights

🚀 As of this release, thanks to @natestemen, **we are officially supporting Python 3.12 and dropping Python 3.9**.

🌉 As part of UnitaryHack 2024, new contributor @NnguyenHTommy **fixed a Qiskit to Cirq gate conversion error** by implementing a fallback mechanism to decompose and transpile the Qiskit circuit into native Cirq gates.

🌀 Another Unitary Hacker, @EmilianoG-byte **added functionality to simulate noise specifically on CNOT and CZ gates when using the Pauli Twilring technique** to symmetrize errors.

🔉 As hinted in last release's spoilers, @purva-thakre has **implemented the noise scaling functionality required for the Layerwise Richardson Extrapolation (LRE) technique**, which allows a more fine-grained control over the amount of noise in circuits compared to the standard unitary folding method.

### ✨ Enhancements
- **Noise Scaling for LRE** (#2347) [@purva-thakre]
- **Issue #2354 Fix - qiskit QFT gates error during conversion** (#2404) [@NnguyenHTommy]
- **Simulate noise for CNOT and CZ gates in Pauli Twirling** (#2397) [@EmilianoG-byte]

### 🛠️ Maintenance and Upkeep Improvements
- Update readme (#2421) [@purva-thakre]
- Update README with Github discussion link (#2419) [@bdg221]
- Update contributing guide (#2382) [@purva-thakre]

### 🧑🏽‍💻 Dev Environment Improvements
- Used trusted publishers for testpypi publishing (#2320) [@natestemen]
- bump min python version for intersphinx map (#2425) [@natestemen]
- **Support python 3.12 and drop 3.9** (#2066) [@natestemen]

#### 📦 Dependency Updates
- Bump qibo from 0.2.8 to 0.2.9 (#2430) [@dependabot[bot]]
- Update qiskit requirement from ~=1.1.0 to ~=1.1.1 (#2414) [@dependabot[bot]]
- Bump pyscf from 2.6.0 to 2.6.2 (#2415) [@dependabot[bot]]


## Version 0.37.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.36.0...v0.37.0))

### Highlights

✨ Stacking quantum error mitigation techniques is a primary area of focus in Mitiq. In this release, @jordandsullivan introduced a **Tutorial on composing Digital Dynamical Decoupling (DDD) and Zero Noise Extrapolation (ZNE)**.

🗒️ **»Download Notebook«**: Users have now the option to download tutorials in Jupyter `.ipynb` format directly from our documentation. We hope this will encourage experimentation with existing tutorials.

🤐 Lastly, a spoiler on what's upcoming: a Request for Comments document by @purva-thakre on adding **Layerwise Noise-Scaling and Multivariate Richardson Extrapolation** has been reviewed and approved. These techniques will soon make their way into Mitiq, stay tuned!


### Enhancements
- Add the RFC LRE link (#2329) [@purva-thakre]
- Example stacking DDD and ZNE (#2345) [@jordandsullivan]
- Add download notebook link in docs (#2363) [@cosenal]
- Add example near-Clifford circuit simulators with links (#2367) [@bdg221]

### Maintenance and upkeep improvements
- Remove qiskit-ibm-provider (#2342) [@andre-a-alves]
- Delete old version of diagram, as unnecessary (#2364) [@jordandsullivan]
- Remove duplicate readme entry in toctree (#2362) [@jordandsullivan]
- Update glossary (#2355) [@purva-thakre]
- Update QSE diagram to call module mitiq.qse, add versions to images, update references to point to v2 image (#2361) [@jordandsullivan]
- Updated docstring for `generate_mirror_circuits` (#2353) [@jordandsullivan]
- Pretty print supported programs enum (#2352) [@cosenal]

### Dev environment improvements
- Document tip for easily generating release note (#2398) [@cosenal]
- Missing unit tests for benchmarking circuits (#2366) [@purva-thakre]

#### 📦 Dependency updates
- Bump pyscf from 2.5.0 to 2.6.0 (#2396) [@dependabot[bot]]
- Update qiskit-aer requirement from ~=0.14.0.1 to ~=0.14.2 (#2393) [@dependabot[bot]]
- Update scipy requirement from <=1.13.0,>=1.10.1 to >=1.10.1,<=1.13.1 (#2380) [@dependabot[bot]]
- Bump qibo from 0.2.7 to 0.2.8 (#2385) [@dependabot[bot]]
- Update qiskit requirement from ~=1.0.2 to ~=1.1.0 (#2369) [@dependabot[bot]]
- Update pennylane-qiskit requirement from ~=0.35.1 to ~=0.36.0 (#2350) [@dependabot[bot]]
- Update pennylane requirement from ~=0.35.1 to ~=0.36.0 (#2349) [@dependabot[bot]]


## Version 0.36.0

([Full Changelog](https://github.com/unitaryfoundation/mitiq/compare/v0.35.0...v0.36.0))

### Highlights

**Support for Qiskit 1.0**: Mitiq now fully supports programs written in Qiskit 1.0, thanks to the contributions of André Alves!

**Enhanced Package Requirements**: We've clarified the requirements for frontend packages. Each frontend is now available as an "extra" within the Mitiq package. For instance, to use Mitiq with Qiskit, simply run:

```bash
pip install mitiq[qiskit]
```
and similarly for all other [supported integrations](https://github.com/unitaryfoundation/mitiq/blob/main/INTEGRATIONS.txt).
This ensures compatibility between all dependency packages required by Mitiq for frontend integration and those in the user's environment.

**Quantum Error Mitigation methods**: Users can now discover the available quantum error mitigation techniques by executing:
```python3
mitiq.qem_methods()
```
This function provides an accessible way to understand the module naming of each technique supported by Mitiq.

Thanks to @andre-a-alves, @cosenal, @jordandsullivan, @mistywahl, @purva-thakre for the PRs in this milestone.

### Enhancements

- Created method to get available QEM methods in mitiq (#2298) [@jordandsullivan]
- Define type for frontend supported programs (#2276) [@cosenal]
- Introduce requirement setup for integrations (#2303) [@cosenal]

### Maintenance and upkeep improvements

- Upgrade Qiskit to 1.0.2 (#2269) [@andre-a-alves]
- Remove unused folding functions (#2289) [@jordandsullivan]
- Use Jupyter cache in gh workflow docs build (#2279) [@cosenal]
- Run linkcheck on 'release' PR workflow (#2332) [@cosenal]

### Dev environment improvements

- Rename master branch to main (#2263) [@jordandsullivan]
- Pre-commit hook with style checks (#2264) [@cosenal]
- Indicate master is under development (#2258) [@natestemen]
- Remove unused devcontainer (#2302) [@cosenal]

#### 📦 Dependency updates

- Bump myst-parser from 2.0.0 to 3.0.1 (#2333) [dependabot[bot]]
- Bump bqskit and scipy (#2262) [dependabot[bot]]
- Bump qibo from 0.2.4 to 0.2.7 (#2268) [dependabot[bot]]

#### 🧑‍💻 Dev Dependency updates

- Update to codecov-action v4 and use token (#2271) [@cosenal]

## Version 0.35.0

In this milestone, we've continued our work to support Qibo by providing a new tutorial, adding related Qibo-conversion functionality to the API-doc, and added Qibo to our main list of supported frontends.
We've also added the capability to use rotated randomized benchmarking circuits as part of the calibrator.
These circuits provide expectation values ranging from 0 to 1 when measuring the probability that the output state is in the ground state.
Having circuits with a wide range of expectation values is an important benchmarking task, and make a great test for finding the correct error mitigation technique/parameters.
If you find any bugs/inconveniences in working with these updates make sure to open an issue so we are able to fix it ASAP!

This release also contains contributions from two new Mitiq contributors, and Unitary Fund team members Alessandro and Jordan!
Welcome both, and looking forward to many more contributions!
Well done making your first contribution so quickly 🏎️💨!

### Commits

- Exit early when circuit type is not supported (#2252) [@cosenal]
- adding rotated randomized benchmarking circuits to calibrator (#2248) [@farlab]
- add qibo + rearrange frontend order (#2249) [@purva-thakre]
- Fix examples link in README.md (#2242) [@jordandsullivan]
- add link to discussions as a way to contribute (#2234) [@natestemen]
- use python 3.11 for RTD (#2231) [@natestemen]
- fix pennylane tutorial (#2232) [@natestemen]
- Add qibo example to docs (#2220) [@francescsabater]
- replace `black`, `flake8`, and `isort` with `ruff` (#2222) [@natestemen]
- pin qibo version (#2221) [@natestemen]
- Add Qibo conversions to documentation (#2214) [@nathanshammah]
- Added warning filter to ignore warning (#2211) [@bdg221]
- remove redundant imports (#2206) [@natestemen]
- Indicate master is under development (#2205) [@natestemen]

#### 📦 Dependency updates

- Update pennylane-qiskit requirement from ~=0.34.0 to ~=0.34.1 (#2198) [@dependabot]
- Update qiskit-ibm-provider requirement from ~=0.8.0 to ~=0.10.0 (#2196) [@dependabot]
- Update pennylane requirement from ~=0.34.0 to ~=0.35.1 (#2227) [@dependabot]
- Update pennylane-qiskit requirement from ~=0.34.1 to ~=0.35.1 (#2225) [@dependabot]
- Bump stimcirq from 1.12.1 to 1.13.0 (#2236) [@dependabot]
- Bump stim from 1.12.1 to 1.13.0 (#2237) [@dependabot]

#### 🧑‍💻 Dev Dependency updates

- Bump all documentation dependencies (#2179) [@dependabot]
- Bump pytest-cov from 4.0.0 to 5.0.0 (#2240) [@dependabot]

## Version 0.34.0

Announcing support for [Qibo](https://qibo.science/), a newly integrated frontend in Mitiq! 📣
Qibo is an "end-to-end open source platform for quantum simulation, self-hosted quantum hardware control, calibration and characterization".

Thank you to new contributor Francesc Sabater for excellent work integrating Qibo and Mitiq!
Thanks also to new contibutor Sam Burdick for fixing our readme.

This release also includes a refactoring of part of the Mitiq shadows module, `mitiq.shadows.classical_postprocessing`, for speed of execution and code readability.

### 📓 Documentation

We've continued to enhance our (legendary!) documentation with:

1. Addition of a security policy document
2. Faster execution of the learning-based PEC tutorial in CI

### Commits

- Add support for qibo circuits (#2102) [@francescsabater]
- Reduce doc build time for learning representations (#2165) [@Misty-W]
- Create SECURITY.md (#2162) [@nathanshammah]
- Fix typo in README.md (#2173) [@smburdick]
- Refactor classical postprocessing in shadows module (#2152) [@natestemen]
- Fix broken link in `combine_rem_zne.md` (#2156) [@Misty-W]

#### 📦 Dependency updates

- Bump openfermion from 1.6.0 to 1.6.1 (#2182) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.68.3 to ~=1.69.0 (#2177) [@dependabot]
- Bump pyscf from 2.4.0 to 2.5.0 (#2176) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.66.0 to ~=1.68.3 (#2175) [@dependabot]
- Update qiskit-aer requirement from ~=0.13.1 to ~=0.13.2 (#2157) [@dependabot]
- Bump pytest from 7.1.3 to 8.0.0 (#2167) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.65.1 to ~=1.66.0 (#2155) [@dependabot]
- Update pennylane-qiskit requirement from ~=0.33.1 to ~=0.34.0 (#2154) [@dependabot]
- Update pennylane requirement from ~=0.33.1 to ~=0.34.0 (#2153) [@dependabot]
- Update qiskit-ibm-provider requirement from ~=0.7.3 to ~=0.8.0 (#2151) [@dependabot]

## Version 0.33.0

Minor update from 0.32.0 to fix a bug 🐛 in the `mitiq.shadows` module where an incorrect index was being used.

### All changes

- Changed to use the split for b_list_shadow (#2137) [@bdg221]

## Version 0.32.0

> Happy holidays, and happy (almost) new year!! ❄️☃️🎄🎊
> This will be our last release of the year, and we'd like to thank everyone who has contributed to Mitiq over the past 12 months.
> We've accomplished so much in the way of error mitigation this year, and we couldn't have done it without the support and time given by the volunteers.
> **Thank you!**

The **calibrator logs have been revamped** for to support result discovery and analysis.
The `Calibrator.run` method now support two options: `flat` and `cartesian` to display the experiment results in either a linear fashion, or grid-like.
Results here have been truncated for brevity.

```py
>>> calibrator.run(log="flat")
┌──────────────────────────┬────────────────────────────────────┬────────────────────────────┐
│ benchmark                │ strategy                           │ performance                │
├──────────────────────────┼────────────────────────────────────┼────────────────────────────┤
│ Type: ghz                │ Technique: ZNE                     │ ✔                          │
│ Num qubits: 2            │ Factory: Linear                    │ Noisy error: 0.04          │
│ Circuit depth: 2         │ Scale factors: 1.0, 2.0, 3.0       │ Mitigated error: 0.02      │
│ Two qubit gate count: 1  │ Scale method: fold_gates_at_random │ Improvement factor: 2.0    │
├──────────────────────────┼────────────────────────────────────┼────────────────────────────┤
│ Type: ghz                │ Technique: ZNE                     │ ✘                          │
│ Num qubits: 2            │ Factory: Linear                    │ Noisy error: 0.04          │
│ Circuit depth: 2         │ Scale factors: 1.0, 3.0, 5.0       │ Mitigated error: 0.0658    │
│ Two qubit gate count: 1  │ Scale method: fold_global          │ Improvement factor: 0.6076 │
├──────────────────────────┼────────────────────────────────────┼────────────────────────────┤
│ Type: ghz                │ Technique: ZNE                     │ ✘                          │
│ Num qubits: 2            │ Factory: Richardson                │ Noisy error: 0.98          │
│ Circuit depth: 33        │ Scale factors: 1.0, 3.0, 5.0       │ Mitigated error: 1.03      │
│ Two qubit gate count: 14 │ Scale method: fold_global          │ Improvement factor: 0.9515 │
└──────────────────────────┴────────────────────────────────────┴────────────────────────────┘
```

```py
>>> calibrator.run(log="cartesian")
┌────────────────────────────────────┬────────────────────────────┬────────────────────────────┐
│ strategy\benchmark                 │ Type: ghz                  │ Type: mirror               │
│                                    │ Num qubits: 2              │ Num qubits: 2              │
│                                    │ Circuit depth: 2           │ Circuit depth: 33          │
│                                    │ Two qubit gate count: 1    │ Two qubit gate count: 14   │
├────────────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Technique: ZNE                     │ ✘                          │ ✘                          │
│ Factory: Richardson                │ Noisy error: 0.03          │ Noisy error: 1.0           │
│ Scale factors: 1.0, 2.0, 3.0       │ Mitigated error: 0.09      │ Mitigated error: 1.03      │
│ Scale method: fold_global          │ Improvement factor: 0.3333 │ Improvement factor: 0.9709 │
├────────────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Technique: ZNE                     │ ✘                          │ ✔                          │
│ Factory: Richardson                │ Noisy error: 0.03          │ Noisy error: 1.0           │
│ Scale factors: 1.0, 3.0, 5.0       │ Mitigated error: 0.0563    │ Mitigated error: 0.97      │
│ Scale method: fold_global          │ Improvement factor: 0.5333 │ Improvement factor: 1.0309 │
├────────────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Technique: ZNE                     │ ✘                          │ ✔                          │
│ Factory: Linear                    │ Noisy error: 0.03          │ Noisy error: 1.0           │
│ Scale factors: 1.0, 3.0, 5.0       │ Mitigated error: 0.0417    │ Mitigated error: 0.9975    │
│ Scale method: fold_global          │ Improvement factor: 0.72   │ Improvement factor: 1.0025 │
└────────────────────────────────────┴────────────────────────────┴────────────────────────────┘
```

**New benchmarking circuits:** `mitiq.benchmarks` now contains a function `generate_random_clifford_t_circuit` which does what it says on the tin.
Special shoutout to new UF team member Farrokh Labib (@farlab) for this contribution.

```py
from mitiq.benchmarks import generate_random_clifford_t_circuit

clifft = generate_random_clifford_t_circuit(
    num_qubits=2,
    num_oneq_cliffords=5,
    num_twoq_cliffords=5,
    num_t_gates=5
)
print(clifft)
# 0: ───────────@───S───T───@───H───T───X───T───T───@───@───────
#               │           │           │           │   │
# 1: ───S───T───@───────────X───────────@───S───────@───X───S───
```

The `Executor.run` method now supports a single circuit instance in addition to a list for ease of use when working with a single circuit.

```diff
- executor.run([circuit])
+ executor.run(circuit)
```

**Faster Tests!** Working on Mitiq has never been easier to develop with a faster (by 36%) test suite.

### 📓 Documentation

This release contains quite a few documentation improvements, including

1. New workflow images to elucidate the workflow for for using the `mitiq.shadows` module (available [here](https://mitiq.readthedocs.io/en/stable/guide/shadows.html))
2. A reorganized API-doc which should be easier to navigate
3. General clean up of the CDR user guide pages

### Commits

- Clarify CDR training docs regarding the use of a markov chain monte carlo (#2130) [@natestemen]
- Update workflow images in documentation (#2034) [@purva-thakre]
- 2115 pauli twirling callibration of expectation estimation shadow needs continue (#2116) [@bdg221]
- Add randomized clifford+T benchmarking circuits. (#2118) [@farlab]
- Frontend/Backend docs clean up (#2124) [@natestemen]
- Speed up tests using mocks (#2126) [@natestemen]
- resolve flaky mirror QV circuit test (#2127) [@natestemen + @misty-w]
- Simplify expectation_estimation_shadow code (#2113) [@bdg221]
- Refactor calibration logs (#2074) [@kozhukalov]
- Organize API-doc (#2104) [@purva-thakre]
- add support for single circuit on exeuctor run method (#2099) [@emilianog-byte]

#### 📦 Dependency updates

- Update qiskit-ibm-provider requirement from ~=0.7.2 to ~=0.7.3 (#2122) [@dependabot]
- Update cirq requirement from <1.3.0,>=1.0.0 to >=1.0.0,<1.4.0 (#2107) [@dependabot]
- Update pennylane requirement from ~=0.32.0 to ~=0.33.1 (#2091) [@dependabot]
- Update pennylane-qiskit requirement from ~=0.32.0 to ~=0.33.0 (#2081) [@dependabot]
- Bump stim/stimcirq from 1.12.0 to 1.12.1 (#2106) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.59.2 to ~=1.64.1 (#2089 + #2094 + #2114 + #2123) [@dependabot]
- Update scipy requirement from <=1.11.3,>=1.5.0 to >=1.5.0,<=1.11.4 (#2095) [@dependabot]

#### 🧑‍💻 Dev Dependency updates

- Bump myst-nb from 0.17.1 to 1.0.0 (#2090) [@dependabot]
- Bump pandas from 2.1.2 to 2.1.3 (#2093) [@dependabot]
- Bump matplotlib from 3.8.0 to 3.8.1 (#2084) [@dependabot]
- Bump isort from 5.12.0 to 5.13.2 (#2120 + #2125) [@dependabot]
- Bump actions/setup-python from 4 to 5 (#2112) [@dependabot]

## Version 0.31.0

Released November 2, 2023

### Summary

This release contains several documentation improvements and some new additions. Quantum subspace expansion (QSE) is added to the user guide (thanks @bubakazouba). Thanks to our first time contributors @dubeyPraY for a new tutorial on using PennyLane and Mitiq in calculating the energy landscape of a simple variational circuit and @kozhukalov for adding the PEC noise level and calculated error to the calibration logs. **We also removed support for python 3.8**.

### All changes

- Second incremental speed up of Mitiq tests [@Misty-W]
- More general conversion decorator and fix conversion bug in PEC (#2064) [@andreamari]
- Added Example for Mitiq in simple landscape for Pennylane (#2048) [@dubeyPraY]
- Indicate under active development on master (#2054) [@Misty-W]
- 2029 update contributing docs.md for references thumbnails viewing rtd build (#2053) [@bdg221]
- Add noise level to the PEC calibration log (#2045) [@kozhukalov]
- 2024 reduce documentation build time after classical shadows added (#2058) [@bdg221]
- Clean up GitHub CI (#2069) [@natestemen]
- remove binder directory (#2071) [@natestemen]
- drop support for python 3.8 (#2068) [nate stemen]
- Include calculated error in calibrator logs (#2038) [@kozhukalov]
- Adds QSE user guide (#1976) [@bubakazouba]

#### Dependency updates

- Bump openfermion from 1.5.1 to 1.6.0 (#2078) [@dependabot]
- Update qiskit-aer requirement from ~=0.12.2 to ~=0.13.0 (#2076) [@dependabot]
- Bump pandas from 2.1.1 to 2.1.2 (#2077) [@dependabot]
- Update qiskit requirement from ~=0.44.2 to ~=0.44.3 (#2073) [@dependabot]
- Update scipy requirement from <=1.11.2,>=1.5.0 to >=1.5.0,<=1.11.3 (#2031) [@dependabot]
- Update qiskit-ibm-provider requirement from ~=0.7.1 to ~=0.7.2 (#2072) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.55.1 to ~=1.59.2 (#2070) [@dependabot]
- Bump pyscf from 2.3.0 to 2.4.0 (#2055) [@dependabot]
- Update qiskit requirement from ~=0.44.1 to ~=0.44.2 (#2041) [@dependabot]
- Update qiskit-ibm-provider requirement from ~=0.7.0 to ~=0.7.1 (#2059) [@dependabot]
- Bump matplotlib from 3.7.3 to 3.8.0 (#2020) [@dependabot]
- Bump pandas from 2.0.3 to 2.1.1 (#2023) [@dependabot]
- Update pennylane-qiskit requirement from ~=0.31.0 to ~=0.32.0 (#1980) [@dependabot]
- Update seaborn requirement from ~=0.12.2 to ~=0.13.0 (#2037) [@dependabot]

## Version 0.30.0

Released October 13, 2023

### Summary

This release contains several documentation improvements and some new additions.
The classical shadows documentation has been improved (including a tutorial!) by @Min-Li. The Pauli Twirling method is added to the user guide (thanks @Aaron-Robertson and @purva-thakre). There is a new tutorial applying both zero-noise extrapolation (ZNE) and Clifford Data Regression (CDR) to quantum simulation, for the 1D Ising chain, in Cirq, by @farzadkianvash, a new contributor! The documentation has been further improved and unified by @Misty-W and @natestemen.

In terms of additions, a new type of benchmark quantum circuits, "rotated" randomized benchmarking (RB) quantum circuits have been added by @Misty-W, for more general benchmarks.

```py
import numpy as np
from mitiq.benchmarks import generate_rotated_rb_circuits

circuits = generate_rotated_rb_circuits(
    n_qubits=2,
    num_cliffords=10,
    theta=2 * np.pi * np.random.Generator.random(),
    trials=100,
)
```

### All changes

- Add a tutorial for simulating Ising 1-D chain with Cirq with ZNE and CDR [@farzadkianvash]
- Add section on quantum noise to user guide (#2036) [@Misty-W]
- New QEM benchmarking method: "rotated" RB circuits (#2028) [@Misty-W]
- Add Pauli Twirling (PT) User Guide (#1848) [@Aaron-Robertson,@purva-thakre]
- Documentation cleanup (#2008) [@natestemen]
- Remove draft workflow from Github Actions (#2019) [@purva-thakre]
- Improve documentation of Classical Shadows (#2026) [@Min-Li]
- Classical Shadows: Add tutorial (#1945) [@Min-Li]

#### Dependency updates

- Update amazon-braket-sdk requirement from ~=1.54.1 to ~=1.55.1 (#2016) [@dependabot]
- Bump matplotlib from 3.7.2 to 3.7.3 (#2011) [@dependabot]

## Version 0.29.0

### Summary

#### Update Pauli Twirling

Thanks to @purva-thakre for updating Mitiq's PT functions, clarifying that PT is a noise tailoring technique and for consolidating utilities to be shared between PT and other techniques.
This release replaces the `execute_with_pt` function with `pauli_twirl_circuit`.

```py
from mitiq.pt.pt import pauli_twirl_circuit

pauli_twirl_circuit(circuit)
```

#### Classical Shadows

Top-level functions and tests for classical shadows estimation are now available in Mitiq.
Congrats @min-li on completing the main functionality for this technique!
Note that documentation for classical shadows estimation is not yet available but coming soon.

```py
from mitiq.shadows.shadows import shadow_quantum_processing, classical_post_processing

shadow_outcomes = shadow_quantum_processing(circuit, executor, num_total_measurements_shadow)
results = classical_post_processing(shadow_outcomes)
```

#### Stim + Mitiq tutorial

Added a tutorial demonstrating a method of combining quantum error mitigation (QEM) and quantum error correction (QEC), reducing the effective logical error rate of the computation.
This tutorial also introduces the use of Mitiq’s ZNE functions with a new backend, the Stim stabilizer simulator.

#### Calibration, Testing, and Documentation

Streamlined formatting of calibration logs, removed redundant test cases, and fixed documentation issues.
Thanks @natestemen for these improvements and for reviewing many of the PRs in this release!

Also, congrats to our new contributor @bdg221 for closing their first Mitiq PR! :tada:

### All changes

- Move functions to utils (#1989) [@purva-thakre]
- remove unused import (#1999) [@natestemen]
- Make robust `Calibrator` logging (#1985) [@natestemen]
- Speed up a few tests (#1996) [@natestemen]
- 1988 contributing doc note for zsh shell (#1997) [@bdg221]
- ZNE Stim tutorial (#1967) [@Misty-W]
- Fix typos in theory section of ddd guide (#1993) [@Misty-W]
- Broken link in docs (#1991) [@purva-thakre]
- Change the main function in Pauli Twirling (#1977) [@purva-thakre]
- remove binder badge and other links binder (#1970) [@andreamari]
- [Classical Shadows 4] Main function (#1921) [@min-li]
- Ensure BQSKit example runs (#1962) [@natestemen]

#### Dependency updates

- Bump stimcirq from 1.11.0 to 1.12.0 (#2000) [@dependabot]
- Bump stim from 1.11.0 to 1.12.0 (#2001) [@dependabot]
- Bump actions/checkout from 3 to 4 (#1994) [@dependabot]
- Update pennylane requirement from ~=0.31.1 to ~=0.32.0 (#1978) [@dependabot]
- Update qiskit-ibm-provider requirement from ~=0.6.3 to ~=0.7.0 (#1982) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.53.4 to ~=1.54.1 (#1972) [@dependabot]
- Update qiskit requirement from ~=0.44.0 to ~=0.44.1 (#1969) [@dependabot]
- Bump scipy + pyscf versions (#1968) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.53.3 to ~=1.53.4 (#1965) [@dependabot]

## Version 0.28.0

### Summary

#### Quantum Subspace Expansion

With the main functionaly implemented, quantum subspace expansion is now available in Mitiq!
The technique requires a sequence of check operators, a Hamiltonian, and an observable in addition to the typical circuit and executor that Mitiq needs to operate.

```py
from mitiq.qse import execute_with_qse

execute_with_qse(circuit, executor, check_operators, code_hamiltonian, observable)
```

This feature is still in flux, and would greatly benefit from further testing.
Do give a try, and [let us know if you have feedback](https://github.com/unitaryfoundation/mitiq/issues/new/)!
More details can be found in our [API-doc](https://mitiq.readthedocs.io/en/latest/apidoc.html#module-mitiq.qse.qse).
Congratulations to @bubakazouba for the great work here.

#### PEC Calibration

Last release we added support to run PEC experiments within the `calibration` module.
This release we made two improvements:

1. Calibration experiments now represent all two-qubit gates by default (previously this was just $\mathrm{C}X$ and $\mathrm{C}Z$ gates.)
2. When running `calibrator.run(log=True)` you will now find results from your PEC pretty-printed alongside any ZNE experiments.

#### Installation

Our core dependencies (NumPy, Cirq, SciPy) are now less tightly specified which means easier installs for users!

#### Robust Shadow Estimation

@Min-Li has been hard at work bringing shadows to Mitiq.
The `shadows` module is not quite ready for use, but you can get a sneak peak of what's to come in the Classical Shadows section of our [API-doc](https://mitiq.readthedocs.io/en/latest/apidoc.html#classical-shadows).

### All changes

- Fix docstring in qse.py (#1944) [@Misty-W]
- Add QSE to API-doc (#1938) [@natestemen]
- fix asv benchmarks (#1937) [@natestemen]
- [Classical Shadows 1] classical postprocessing (#1908) [@min-li]
- Add Calibration logging for PEC (#1873) [@Misty-W]
- [Classical Shadows 1] utils for shadows and unit test (#1907) [@min-li]
- Clean up types (#1825) [@natestemen]
- Relax test condition for `test_execute_with_pauli_twirling` (#1931) [@Misty-W]
- loosen remaining core dependency versions (#1917) [@natestemen]
- [Classical Shadows] quantum processing and test (#1906) [@min-li]
- Make PEC calibration support all multi-qubit-gate (#1881) [@YuNariai]
- Implements high-level functions for QSE (#1902) [@bubakazouba]
- Temporary fix for documentation problem (#1927) [@andreamari]
- Adding layerwise folding as tutorial to mitiq. (#1894) [@vprusso]
- Fix docstring of initialized_depolarized_noise (#1919) [@andreamari]
- Update broken URLs (#1912) [@min-li]
- indicate under active development on master (#1900) [@natestemen]

#### Dependency updates

- Update pennylane requirement from ~=0.30.0 to ~=0.31.0 (#1888) [@dependabot]
- Update cirq requirement from ~=1.1.0 to ~=1.2.0 (#1922) [@dependabot]
- Update qiskit requirement from ~=0.43.3 to ~=0.44.0 (#1935) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.51.0 to ~=1.52.0 (#1933) [@dependabot]
- Update qiskit-ibm-provider requirement from ~=0.6.1 to ~=0.6.2 (#1932) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.50.0 to ~=1.51.0 (#1928) [@dependabot]
- Update qiskit requirement from ~=0.43.2 to ~=0.43.3 (#1925) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.49.1 to ~=1.50.0 (#1926) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.49.0 to ~=1.49.1 (#1916) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.46.0 to ~=1.49.0 (#1915) [@dependabot]

## Version 0.27.0

### Summary

Highlights from this release include adding new benchmark quantum circuits: Mirror Quantum Volume Circuits (@purva-thakre) and adding PEC as technique supported by calibration (@Misty-W). After approval of the related RFC on quantum subspace expansion technique, the first utils have been added (@bubakazouba). Other improvements include a new tutorial on quantum many body scars (@DHuybrechts); issues solved during unitaryHACK such as improvement to the cost estimation for Calibrator (@YuNariai), Qiskit Upgrade and Deprecation Warnings (@andre-a-alves), and a new function to register user defined Mitiq converters (@Aaron-Robertson).

### All changes

- Draft workflow to run change specific tests (#1809) @Aaron-Robertson
- Improve twirling test (#1831) @andreamari
- Add banner to docs (#1834) @natestemen
- Supports observable multiplication with observable and PauliString @bubakazouba
- Add tutorial on Quantum Many Body Scars with ZNE @DHuybrechts
- Use latest copyright notice (#1892) @natestemen
- Fix frozen modules (#1879) @purva-thakre
- Update links for accepted RFCs (#1884) @purva-thakre
- Ensure execute_with_rem works with Executor object (#1877) @natestemen @Misty-W
- Remove unitaryHACK banner (#1875)@natestemen
- Braket example on mitigating the energy landscape of a variational CI @deji725
- Improve cost estimation for Calibrator (#1863) @YuNariai
- Mirror Quantum Volume Circuits (#1838) @purva-thakre
- Update amazon-braket-sdk requirement from ~=1.41.0 to ~=1.42.1 (#1870) @dependabot committed last month
- Clean up global, isolate tests, and fix mock module (#1864) @Aaron-Robertson
- Adds subspace expansion utils. (#1859) @bubakazouba
- Qiskit Upgrade and Deprecation Warnings (#1847) @andre-a-alves
- Adding PEC as technique supported by calibration (#1845) @Misty-W @andreamari
- Update wording now that event has started (#1860) @natestemen
- Add function to register user defined Mitiq converters (#1850) @Aaron-Robertson
- Removed Windows note (#1857) @andre-a-alves
- Update GitHub link (#1854) @andre-a-alves
- Include current year in copyright notice (#1852) @andre-a-alves
- Make sure PEC preserves measurement gates (#1844) @andreamari
- Add mypy to style guidelines (#1841) @purva-thakre
- Update calibration tutorial (#1840) @Misty-W
- **Dependabot updates:**
- Update pyquil requirement from ~=3.5.0 to ~=3.5.1 @dependabot
- Update amazon-braket-sdk requirement from ~=1.38.0 to ~=1.38.1 (#1829) @dependabot
- Update amazon-braket-sdk requirement from ~=1.45.0 to ~=1.46.0 (#1893) @dependabot
- Update amazon-braket-sdk requirement from ~=1.44.0 to ~=1.45.0 (#1889) @dependabot
- Update qiskit requirement from ~=0.43.1 to ~=0.43.2 (#1890) @dependabot
- Update amazon-braket-sdk requirement from ~=1.43.0 to ~=1.44.0 (#1887) @dependabot
- Update pennylane-qiskit requirement from ~=0.30.1 to ~=0.31.0 (#1886) @dependabot
- Update amazon-braket-sdk requirement from ~=1.42.1 to ~=1.43.0 (#1883) @dependabot
- Update qiskit-ibm-provider requirement from ~=0.6.0 to ~=0.6.1 (#1872) @dependabot
- Update pyquil requirement from ~=3.5.2 to ~=3.5.4 (#1867) @dependabot
- Update qiskit requirement from ~=0.43.0 to ~=0.43.1 (#1868) @dependabot
- Update amazon-braket-sdk requirement from ~=1.40.0 to ~=1.41.0 (#1865) @dependabot
- Update pennylane-qiskit requirement from ~=0.29.0 to ~=0.30.1 (#1824) @dependabot
- Update amazon-braket-sdk requirement from ~=1.38.1 to ~=1.40.0 (#1849) @dependabot
- Update pyquil requirement from ~=3.5.1 to ~=3.5.2 (#1856) @dependabot

## Version 0.26.0

### Summary

Highlights from this release include functions for applying Pauli Twirling of CNOT and CZ gates, support for noise scaling by circuit layer in ZNE, functions to generate Quantum Phase Estimation benchmarking circuits, and a new example composing two Mitiq techniques: REM and ZNE.
Special thanks to UF Ambassadors **Purva Thakre** and **Aaron Robertson** for their contributions to this release!

The use of the Pauli Twirling module is demonstrated in the following code cell<sup>\*</sup>.

```py
from mitiq import pt
twirled_value = pt.execute_with_pauli_twirling(circuit, expval_executor)
```

<sup>\*</sup>Thorough testing and documentation of Pauli Twirling to follow in future releases.
If any bugs or inconsistencies are encountered, please [open an issue](https://github.com/unitaryfoundation/mitiq/issues/new).

### All changes

- CNOT twirling (#1802) [@natestemen]
- Compose REM + ZNE in Mitiq (#1745) [@Misty-W]
- bump nbsphinx (#1821) [@natestemen]
- Update pennylane requirement from ~=0.29.1 to ~=0.30.0 (#1819) [@dependabot[bot]]
- Update amazon-braket-sdk requirement from ~=1.37.1 to ~=1.38.0 (#1817) [@dependabot[bot]]
- Update amazon-braket-sdk requirement from ~=1.37.0 to ~=1.37.1 (#1804) [@dependabot[bot]]
- Support noise scaling by layer (#1767) [@vprusso]
- Remove version specificity from codecov/codecov-action (#1801) [@dependabot[bot]]
- Update pyquil requirement from ~=3.4.1 to ~=3.5.0 (#1793) [@dependabot[bot]]
- use 1.7.0 link for grove (#1803) [@natestemen]
- Remove allcontributors (#1791) [@natestemen]
- Update qiskit tutorial with ddd functions (#1762) [@Misty-W]
- Add calibration workflow to README (#1778) [@natestemen]
- Make W-state circuits available to use in `Calibration` (#1792) [@Misty-W]
- W State and QPE Benchmarking circuits in API Doc (#1785) [@purva-thakre]
- Update pyquil requirement from ~=3.4.0 to ~=3.4.1 (#1786) [@dependabot[bot]]
- Uprade python support to 3.11 (#1663) [@natestemen]
- Update pyquil requirement from ~=3.3.5 to ~=3.4.0 (#1784) [@dependabot[bot]]
- Bump codecov/codecov-action from 3.1.1 to 3.1.2 (#1781) [@dependabot[bot]]
- Quantum Phase Estimation Benchmarking Circuit (#1775) [@purva-thakre]

## Version 0.25.0

### Summary

Highlights from this release include a bug fixed in DDD, extended documentation for identity insertion as a noise scaling technique, new results from testing DDD on IBMQ hardware, a new function `mitiq.benchmarks.w_state_circuits.generate_w_circuit` to generate W-state circuits, and a finalized calibration API.

**Breaking Changes**: The `force_run_all` option for the [`evaluate`](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.executor.executor.Executor.evaluate) method defined on `Executor` objects now defaults to True.

### All changes

- Calibration API simplifications (#1763) [@natestemen]
- W state Benchmarking Circuit (#1723) [@purva-thakre]
- Consider id as slack in ddd circuit mask (#1744) [@Aaron-Robertson]
- Remaining Stuff related to Identity Insertion Scaling in Docs (#1759) [@purva-thakre]
- Add calibration tutorial (#1756) [@nathanshammah]
- Set default force_run_all as True in PEC (#1755) [@andreamari]
- Update amazon-braket-sdk requirement from ~=1.36.3 to ~=1.36.4 (#1765) [@dependabot]
- Make the Calibrator multi-platform (#1748) [@andreamari]
- Update amazon-braket-sdk requirement from ~=1.36.2 to ~=1.36.3 (#1761) [@dependabot]
- Update DDD Qiskit tutorial with hardware experiments (#1751) [@Misty-W]
- Bump pyscf from 2.1.1 to 2.2.0 (#1753) [@dependabot]
- Update pyquil requirement from ~=3.3.3 to ~=3.3.4 (#1750) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.36.1 to ~=1.36.2 (#1749) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.35.5 to ~=1.36.1 (#1742) [@dependabot]
- Fix DDD typo in README (#1743) [@natestemen]
- Update pennylane requirement from ~=0.29.0 to ~=0.29.1 (#1741) [@dependabot]
- Set `master` in dev mode (#1736) [@Misty-W]

## Version 0.24.0

Highlights of this release include refactoring of parts of the PEC module, improvements to the Calibration data and settings structures,
completion of the REM section of the user guide, and the publishing of a Mitiq tutorial first presented as a lab exercise at the [SQMS/GGI 2022 Summer School on Quantum Simulation of Field Theories](https://sqmscenter.fnal.gov/opportunities/summer-schools/#2022).
Special thanks to UF Ambassadors **Amir Ebrahimi** and **Purva Thakre** for their contributions to this release!

**Breaking changes:** The class `NoisyOperation` is deprecated and removed from Mitiq. Moreover the initialization arguments of the `OperationRepresentation` class changed. Please check the associated API-docs and the [PEC section](https://mitiq.readthedocs.io/en/stable/guide/pec.html) of the docs for more details.

- Update pennylane-qiskit requirement from ~=0.28.0 to ~=0.29.0 (#1733) [@dependabot[bot]]
- Update pennylane requirement from ~=0.28.0 to ~=0.29.0 (#1732) [@dependabot[bot]]
- Refactoring of NoisyOperation, NoisyBasis, OperationRepresentation (#1712) [@andreamari]
- Add Summer School Notebook to Docs (#1720) [@purva-thakre]
- Fix mypy errors on #1728 (#1729) [@Misty-W]
- Update settings.py (#1728) [@Misty-W]
- Improve default `ZNESettings` for calibration (#1721) [@Misty-W]
- Update qiskit requirement from ~=0.41.0 to ~=0.41.1 (#1727) [@dependabot[bot]]
- Bump actions/setup-python from 3 to 4 (#1726) [@dependabot[bot]]
- remove pdf doc build (#1725) [@natestemen]
- REM: Update the fifth section of the docs (#1718) [@amirebrahimi]
- [calibration] usability improvements (#1714) [@natestemen]
- Simplify noisy operation (#1713) [@andreamari]
- Calibration results data structure refactor (#1706) [@natestemen]
- Update scipy requirement from ~=1.10.0 to ~=1.10.1 (#1719) [@dependabot[bot]]
- Update amazon-braket-sdk requirement from ~=1.35.4 to ~=1.35.5 (#1717) [@dependabot[bot]]
- Bump mypy from 0.982 to 1.0.0 (#1707) [@dependabot[bot]]
- Remove unnecessary circuit conversion in PEC (#1710) [@Misty-W]
- Update amazon-braket-sdk requirement from ~=1.35.3 to ~=1.35.4 (#1711) [@dependabot[bot]]
- Ignore a temporarely broken but correct DOI link (#1705) [@andreamari]
- Upgrade to qiskit 0.41.0 manually after failed dependabot PR (#1703) [@Misty-W]
- Update README.md (#1702) [@andreamari]
- Revert "Update qiskit requirement from ~=0.40.0 to ~=0.41.0 (#1697)" (#1701) [@Misty-W]
- Update qiskit requirement from ~=0.40.0 to ~=0.41.0 (#1697) [@dependabot[bot]]
- Set version in dev mode (#1700) [@andreamari]
- advertise the fact that we have DDD examples now (#1698) [@natestemen]

## Version 0.23.0

The main improvements introduced in this release are:

- A significant refactoring of the Mitiq [calibration module](https://mitiq.readthedocs.io/en/latest/guide/calibrators.html). We generalized the `Settings`
  object, which is now able to generate a more general list of `BenchmarkProblem` objects (wrapping circuits and ideal results) and a list of `Strategy` objects
  representing the error mitigation strategies to compare. We also improved how the optimal `Strategy` is determined. Specifically, we now average over `BenchmarkProblems` to reduce fluctuations and spurious results.
  We remark that the `mitiq.calibration` module is very new and quickly evolving. Therefore further significant breaking changes are likely to happen in future releases.

- A non-trivial refactoring of the REM module. We changed the underlying workflow of the technique which is now applied directly to executors, instead of applying REM during the evaluation of expectation values. Expectation values can still be mitigated as usual with `execute_with_rem` but mitigated executors can now return raw `MeasurementResult` objects (bitstrings).

- We also significantly extended the REM documentation with new and informative sections. Special thanks to @amirebrahimi and @nickdgardner for their high-quality and useful contributions!

- We now have 2 tutorials focused on digital dynamical decoupling (DDD)---one for Cirq and one for Qiskit---both showing an improvement for a theoretical highly-correlated noise model. Moreover, the Qiskit tutorial on DDD is a useful starting point for testing the technique on real hardware.

### All changes

- Use closest probability distribution in REM (#1688) [@andreamari]
- REM: Write-up third section of docs (#1693) [@nickdgardner]
- Improve calibration method parameter selection (#1682) [@Misty-W]
- Add References about REM (#1691) [@nathanshammah]
- REM: mitigate executors as a first step and expectation values at a second step. (#1678) [@andreamari]
- Draft for glossary of QEM concepts in docs (#1647) [@nickdgardner]
- myst -> md (#1689) [@natestemen]
- use executor.run in rem docs (#1690) [@andreamari]
- REM: Write-up fourth section of docs (#1680) [@amirebrahimi]
- upgrade `_run` to a public function (#1684) [@natestemen]
- Update pyquil requirement from ~=3.3.2 to ~=3.3.3 (#1686) [@dependabot[bot]]
- Update qiskit requirement from ~=0.39.5 to ~=0.40.0 (#1687) [@dependabot[bot]]
- Calibration improvements (#1676) [@natestemen]
- Further updates related to the measurement result class (#1673) [@andreamari]
- DDD on IBMQ backends (#1665) [@Misty-W]
- Update amazon-braket-sdk requirement from ~=1.35.2 to ~=1.35.3 (#1674) [@dependabot[bot]]
- Update qiskit requirement from ~=0.39.4 to ~=0.39.5 (#1675) [@dependabot[bot]]
- Use `md` extension for myst markdown files (#1671) [@natestemen]
- add H2 image (#1677) [@andreamari]
- Use source svg files instead of exported pngs (#1651) [@amirebrahimi]
- Improve and modify MeasurementResult class (#1670) [@andreamari]
- Update pennylane requirement from ~=0.27.0 to ~=0.28.0 (#1643) [@dependabot[bot]]
- Update cirq requirement from ~=1.0.0 to ~=1.1.0 (#1644) [@dependabot[bot]]
- loosen numpy requirements (#1672) [@natestemen]
- clean up latex (#1669) [@natestemen]
- Fix executor and update wording of DDD tutorial (#1655) [@Misty-W]
- Update scipy requirement from ~=1.9.3 to ~=1.10.0 (#1657) [@dependabot[bot]]
- update link (#1661) [@natestemen]
- Update amazon-braket-sdk requirement from ~=1.35.1 to ~=1.35.2 (#1656) [@dependabot[bot]]
- Update numpy requirement from ~=1.24.0 to ~=1.24.1 (#1652) [@dependabot[bot]]

## Version 0.22.0

### Summary

In this release we focused on improving our documentation with three new examples, new REM docs, and navigation improvements, along with some bug fixes.
This release also adds a new module `calibration` which allows one to run a series of experiments to see what error mitigation parameters will work best for their particular setup.
This feature is still under active development.

Many thanks to @amirebrahimi for his continued work on making readout error mitigation available, and usable in Mitiq.

Thanks to everyone who contributed to Mitiq this year!
It's been a great time for error mitigation, and we look forward to continuing to grow Mitiq in the new year!
Happy holidays! 🎄🎉🎊

### All changes

- Digital Dynamical Decoupling Tutorial on Mirror Circuits (#1645) [@Misty-W]
- Fix bug for Qiskit circuits with idle qubits (#1646) [@andreamari]
- Calibration core concepts docs (#1648) [@natestemen]
- Calibration prototype (#1614) [@natestemen]
- Noise scaling tutorial (identity insertion/folding) (#1633) [@natestemen]
- Fill in second sub-section of REM docs (#1630) [@amirebrahimi]
- Add REM docs sections and fill out the first section (#1629) [@amirebrahimi]
- Update pennylane-qiskit requirement from ~=0.27.0 to ~=0.28.0 (#1641) [@dependabot]
- Revert "Update pennylane requirement from ~=0.27.0 to ~=0.28.0 (#1640)" (#1642) [@natestemen]
- Update pennylane requirement from ~=0.27.0 to ~=0.28.0 (#1640) [@dependabot]
- Update numpy requirement from ~=1.23.5 to ~=1.24.0 (#1638) [@dependabot]
- Update mitiq build badge display name (#1636) [@Misty-W]
- Fix build badge (#1635) [@andreamari]
- Contributing/dev tasks docs improvements (#1615) [@natestemen]
- Fix extrapolations for ExpFactory in special cases (#1631) [@Misty-W]
- docs: add Misty-W as a contributor for ideas, test, and 2 more (#1632) [@allcontributors]
- Update learning depolarizing noise tutorial to avoid long execution time (#1628) [@Misty-W]
- Format examples' titles and list (#1622) [@nathanshammah]
- Update qiskit requirement from ~=0.39.3 to ~=0.39.4 (#1625) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.34.3 to ~=1.35.1 (#1626) [@dependabot]
- Add list of accepted RFCs to docs (#1623) [@nathanshammah]
- REM: Utility method maintenance (#1610) [@amirebrahimi]
- Adding PEC simulator mitiq tutorial (#1612) [@vprusso]

## Version 0.21.0

### Summary

This release officially adds support for the learning-based PEC sub-technique which is now
[fully documented](https://mitiq.readthedocs.io/en/latest/guide/pec-3-options.html#applying-learning-based-pec) and ready to be applied by
Mitiq users.
We are still assessing the stability of this new sub-technique, so if you notice any bugs, please let us know by opening
[issues](https://github.com/unitaryfoundation/mitiq/issues) on GitHub.

Functions to apply Readout Error Mitigation (REM) are also introduced in this release, special thanks to Amir Ebrahimi for this contribution!

Also, the [noise scaling by identity insertion](https://mitiq.readthedocs.io/en/latest/guide/zne-3-options.html#identity-scaling)
method is included in the ZNE section of the user guide.
Special thanks to Purva Thakre for this contribution!

During the release cycle we accepted the
[RFC for implementation of calibration tools (Solution 1)](https://github.com/unitaryfoundation/mitiq/issues/1578).
We also completed a prototype of this approach, which will be released in a future version of Mitiq.

In addition, this release adds support for qubit-independent representations for PEC, along with bug fixes and minor dependency upgrades.

### All changes

- Cleanup (some) Deprecation Warnings (#1556) [@purva-thakre]
- Update qiskit requirement from ~=0.39.2 to ~=0.39.3 (#1611) [@dependabot]
- Documentation for Identity Scaling (#1573) [@purva-thakre]
- Bump flake8 from 5.0.4 to 6.0.0 (#1609) [@dependabot]
- Techniques intro documentation (#1603) [@nathanshammah]
- Update pennylane requirement from ~=0.26.0 to ~=0.27.0 (#1590) [@dependabot]
- Enable user to define qubit-independent representations (#1565) [@Misty-W]
- Refactor generate_inverse_confusion_matrix (#1607) [@amirebrahimi]
- REM implementation (#1449) [@amirebrahimi]
- Update numpy requirement from ~=1.23.4 to ~=1.23.5 (#1600) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.34.1 to ~=1.34.3 (#1599) [@dependabot]
- remove skipped test (#1598) [@natestemen]
- Write user guide section for learning-based PEC (#1395) [@Misty-W]
- Raise error when inserting DDD sequence with midcircuit measurement (#1586) [@natestemen]
- use \prod instead of \Pi documentationImprovements or additions to documentation (#1592) [@natestemen]
- Update amazon-braket-sdk requirement from ~=1.34.0 to ~=1.34.1 (#1595) [@dependabot]
- Learn biased noise parameters (#1510) [@Misty-W]
- Update qiskit requirement from ~=0.39.0 to ~=0.39.1 (#1579) [@dependabot]
- Update qiskit requirement from ~=0.39.1 to ~=0.39.2 (#1581) [dependabot]
- Update amazon-braket-sdk requirement from ~=1.31.1 to ~=1.33.1 (#1585) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.33.1 to ~=1.33.2 (#1588) [@dependabot]
- Update pennylane-qiskit requirement from ~=0.24.0 to ~=0.27.0 (#1591) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.33.2 to ~=1.34.0 (#1589) [@dependabot]
- color output from pytest on CI (#1587) [@natestemen]
- Implement conversions of IonQ native gates for Braket (#1525) [@yitchen-tim]
- use python 3.10 for asv (#1580) [@natestemen]

## Version 0.20.0

### Summary

This milestone focused on updating our support for numpy (1.23), adding a tutorial demonstrating the learning-based PEC workflow, and scoping out what device/noise calibration might look like as part of Mitiq.
Additionally identity insertion has been added as a noise-scaling technique available for mitigation protocols such as zero-noise extrapolation.
Expect more documentation of this feature in future releases.
Big thanks to @purva-thakre for getting this in Mitiq!

There are also some minor bug fixes, documentation updates, and a new example contributed by @nickdgardner as well!

### All changes

- Port BQSKit example to 1.0 release (#1557) [@natestemen]
- Add Qiskit example on mitigating the energy landscape of a variational circuit (#1551) [@nickdgardner]
- Bump pytest-xdist[psutil] from 2.5.0 to 3.0.2 (#1569) [@dependabot]
- remove badges from contributing to docs doc (#1571) [@natestemen]
- Warn when unsupported gate fidelity is passed (#1542) [@natestemen]
- remove gh-pages deploy action (#1566) [@natestemen]
- First demo of learning function (#1514) [@Misty-W]
- remove stale bot (#1558) [@natestemen]
- Update scipy requirement from ~=1.9.2 to ~=1.9.3 (#1561) [@dependabot]
- Docs infrastructure improvements (#1559) [@natestemen]
- Identity Insertion Scaling (#1442) [@purva-thakre]
- Update scipy requirement from ~=1.9.1 to ~=1.9.2 (#1548) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.31.0 to ~=1.31.1 (#1549) [@dependabot]
- Update qiskit requirement from ~=0.38.0 to ~=0.39.0 (#1550) [@dependabot]
- Update numpy requirement from ~=1.23.3 to ~=1.23.4 (#1553) [@dependabot]
- Revert "[temporary] ignore errors from first-interaction (#1532)" (#1555) [@natestemen]
- allow all tests to run if one fails (#1554) [@natestemen]
- Update numpy requirement from ~=1.21.6 to ~=1.23.3 (#1486) [@dependabot]
- Update black requirement from ~=22.8 to ~=22.10 (#1544) [@dependabot]
- Update pydata-sphinx-theme requirement from ~=0.10.1 to ~=0.11.0 (#1545) [@dependabot]
- add robots.txt 🤖 (#1543) [@natestemen]
- Set master in dev mode (#1536) [@andreamari]
- Update mypy requirement from ~=0.981 to ~=0.982 (#1537) [@dependabot]
- Update pyquil requirement from ~=3.3.1 to ~=3.3.2 (#1539) [@dependabot]

## Version 0.19.0

### Summary

With this release Mitiq supports most recent versions of Python: `3.8`, `3.9` and `3.10`!
We drop support for Python `3.7`.

Mitiq is now compatible with Numpy `1.21.6`. Different versions of NumPy may not work properly.

Another important update is the addition of new tools for applying learning-based PEC!
This release introduces a function for learning depolarizing noise representations from Clifford circuit data.
Read more in our updated [API-doc](https://mitiq.readthedocs.io/en/latest/apidoc.html#module-mitiq.pec.representations.biased_noise).

Special thanks to the external contributors @yitchen-tim, @amirebrahimi and @isaac-gs!

### All Changes

- Implement conversions of IonQ native gates for Braket (#1525) [@yitchen-tim]
- [temporary] ignore errors from first-interaction (#1532) [@natestemen]
- Update sphinx-autodoc-typehints requirement from ~=1.12.0 to ~=1.19.4 (#1515) [@dependabot[bot]]
- Update pytest-cov requirement from ~=3.0.0 to ~=4.0.0 (#1521) [@dependabot[bot]]
- Update myst-nb requirement from ~=0.16.0 to ~=0.17.1 (#1529) [@dependabot[bot]]
- replace deprecated `execution_excludepatterns` (#1530) [@natestemen]
- Support python 3.10 (#1504) [@natestemen]
- Fix outdated link in README (2021 -> 2022) (#1528) [@andreamari]
- Add learning-based PEC subsection in API doc (#1524) [@nathanshammah]
- remove `print` calls (#1519) [@natestemen]
- BQSKit tutorial (#1489) [@natestemen]
- Update pyquil requirement from ~=3.3.0 to ~=3.3.1 (#1516) [@dependabot[bot]]
- Update scipy requirement from ~=1.7.3 to ~=1.9.1 (#1467) [@dependabot[bot]]
- Update mypy requirement from ~=0.971 to ~=0.981 (#1511) [@dependabot[bot]]
- (partially) upgrade numpy (#1501) [@natestemen]
- Update amazon-braket-sdk requirement from ~=1.30.2 to ~=1.31.0 (#1512) [@dependabot[bot]]
- Add whitepaper publication info/links (#1456) [@natestemen]
- Support python 3.9 (#1505) [@natestemen]
- Learning function for biased noise (#1358) [@Misty-W]
- Update amazon-braket-sdk requirement from ~=1.30.1 to ~=1.30.2 (#1506) [@dependabot[bot]]
- Bump actions/stale from 5 to 6 (#1502) [@dependabot[bot]]
- Update amazon-braket-sdk requirement from ~=1.30.0 to ~=1.30.1 (#1499) [@dependabot[bot]]
- Restore the titles of the examples in the Mitiq docs (#1497) [@andreamari]
- Update amazon-braket-sdk requirement from ~=1.29.4 to ~=1.30.0 (#1492) [@dependabot[bot]]
- Bump codecov/codecov-action from 3.1.0 to 3.1.1 (#1496) [@dependabot[bot]]
- removed unused lindblad doc (#1494) [@natestemen]
- Update pennylane requirement from ~=0.25.1 to ~=0.26.0 (#1495) [@dependabot[bot]]
- [MCMC 1/3] Moving clifford utils for cdr to own files (#1480) [@isaac-gs]
- Update qiskit requirement from ~=0.37.2 to ~=0.38.0 (#1490) [@dependabot[bot]]
- docs: add amirebrahimi as a contributor for code, test, doc (#1487) [@allcontributors[bot]]
- Move MeasurementResult into \_typing to fix circular dependency in #1449 (#1484) [@amirebrahimi]
- Update amazon-braket-sdk requirement from ~=1.29.3 to ~=1.29.4 (#1483) [@dependabot[bot]]

## Version 0.18.0

### Summary

This release cycle focused on review and approval of two RFCs, one for Readout Error Mitigation (REM) [#1387](https://github.com/unitaryfoundation/mitiq/issues/1387) and one for Identity insertion noise scaling [#335](https://github.com/unitaryfoundation/mitiq/issues/335) (not listed as PRs). It also includes bug fixes and minor dependency upgrades.

### All Changes

- Update amazon-braket-sdk requirement from ~=1.29.2 to ~=1.29.3 (#1481) [@dependabot]
- Update black requirement from ~=22.6 to ~=22.8 (#1477) [@dependabot]
- docs: add Aaron-Robertson as a contributor for ideas, code, doc (#1464) [@allcontributors]
- docs: add 1ucian0 as a contributor for infra, code (#1461) [@allcontributors]
- Update amazon-braket-sdk requirement from ~=1.29.0 to ~=1.29.2 (#1459) [@dependabot]
- Update qiskit requirement from ~=0.37.1 to ~=0.37.2 (#1458) [@dependabot]
- Update sphinxcontrib-bibtex requirement from ~=2.4.2 to ~=2.5.0 (#1457) [@dependabot]
- Mid circuit measurement fixed for ddd (#1446) [@Aaron-Robertson]
- Update pennylane requirement from ~=0.24.0 to ~=0.25.1 (#1444) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.28.1 to ~=1.29.0 (#1437) [@dependabot]
- Update pyquil requirement from ~=3.2.1 to ~=3.3.0 (#1436) [@dependabot]
- remove patch release docs (#1435) [@natestemen]
- 0.18.0 in dev (#1433) [@natestemen]

## Version 0.17.1

### Summary

This patch release includes support for the latest versions of Qiskit (0.37.1), Cirq (1.0.0), and Pyquil (3.2.1), along with other minor dependency upgrades and bug fixes.

### All Changes

- Update pyquil requirement from ~=3.0.0 to ~=3.2.1 (#1425) [@dependabot]
- Update cirq requirement from ~=0.15.0 to ~=1.0.0 (#1402) [@dependabot]
- Add fold methods as parameters to benchmarks (#1374) [@natestemen]
- Updates tests and docstrings for cirq utils (#1371) [@purva-thakre]
- Update amazon-braket-sdk requirement from ~=1.27.1 to ~=1.28.1 (#1430) [@dependabot]
- Bump openfermion from 1.5.0 to 1.5.1 (#1429) [@dependabot]
- Update flake8 requirement from ~=5.0.3 to ~=5.0.4 (#1423) [@dependabot]
- Update flake8 requirement from ~=4.0.1 to ~=5.0.3 (#1419) [@dependabot]
- Support for Qiskit 0.37 (#1421) [@1ucian0]
- Update amazon-braket-sdk requirement from ~=1.26.1 to ~=1.27.1 (#1417) [@dependabot]
- Bump openfermion from 1.3.0 to 1.5.0 (#1412) [@dependabot]
- Update amazon-braket-sdk requirement from ~=1.26.0 to ~=1.26.1 (#1405) [@dependabot]
- Update mypy requirement from ~=0.961 to ~=0.971 (#1406) [@dependabot]
- Update release docs (#1404) [@natestemen]
- Update amazon-braket-sdk requirement from ~=1.25.2 to ~=1.26.0 (#1403) [@dependabot]
- update version.txt to dev (#1398) [@natestemen]
- Prevent idle qubits with qiskit barriers from being lost after conversions (#1369) [@andreamari]
- Add custom RTD favicon (#1390) [@natestemen]
- Update python-rapidjson requirement from <=1.6 to <1.8 (#1389) [@dependabot]

## Version 0.17.0

### Summary

This release includes contributions from UnitaryHACK 2022! 🎉
We had 3 merged pull requests, and a fourth is looking to be merged in the next milestone.
Along with the great contributions from hackers, this release focused on expanding our ZNE examples to other frontends (Cirq, Braket, and Pennylane), building out learning-based PEC, and improving our benchmarking infrastructure.

### All Changes

- Support the latest Cirq version (0.15.0) (@andreamari, gh-1379)
- add ZNE example using pennylane (@natestemen, gh-1384)
- Add ZNE example with Braket and IonQ (@andreamari, gh-1363)
- docs: add natestemen as a contributor for infra, doc, test, code (allcontributors, gh-1383)
- remove comma (@natestemen, gh-1381)
- add myself to authors (@natestemen, gh-1372)
- Deploy asv benchmark data to web (@natestemen, gh-1354)
- Batched mitigate (@nylewong, gh-1286)
- Update black requirement from ~=22.3 to ~=22.6 (@dependabot, gh-1370)
- Add Error Mitigation example with Cirq (@Misty-W, gh-1351)
- Update amazon-braket-sdk requirement from ~=1.25.1 to ~=1.25.2 (@dependabot, gh-1367)
- Update pennylane-qiskit requirement from ~=0.23.0 to ~=0.24.0 (@dependabot, gh-1361)
- Update Mitiq paper code blocks based on paper review (@andreamari, gh-1357)
- Update pennylane requirement from ~=0.23.1 to ~=0.24.0 (@dependabot, gh-1362)
- Update amazon-braket-sdk requirement from ~=1.25.0 to ~=1.25.1 (@dependabot, gh-1359)
- Improving the H2 example (@obliviateandsurrender, gh-1337)
- Test on windows and mac on each PR and add openfermion to dev requirements (@andreamari, gh-1348)
- Loss function for learning biased noise (@Misty-W, gh-1340)
- Modify docstrings, documentation to use citations from refs.bib [unitaryhack] (@wingcode, gh-1325)
- [unitaryhack] v2 Updated molecular hydrogen tutorial OpenFermion (@lockwo, gh-1349)
- Update amazon-braket-sdk requirement from ~=1.24.0 to ~=1.25.0 (@dependabot, gh-1345)
- Bump actions/setup-python from 3 to 4 (@dependabot, gh-1344)
- [unitaryhack] Add thumbnails for examples (@obliviateandsurrender, gh-1338)
- Revert "[unitaryhack] Updated molecular hydrogen tutorial" (@andreamari, gh-1347)
- [unitaryhack] Updated molecular hydrogen tutorial (@lockwo, gh-1296)
- Update mypy requirement from ~=0.960 to ~=0.961 (@dependabot, gh-1342)
- Merge pull request #1320 from unitaryfund/nts-contrib-docs (@natestemen)

## Version 0.16.0 (June 3rd, 2022)

### Summary

This release officially adds support for the digital dynamical decoupling (DDD) technique which is now [fully documented](https://mitiq.readthedocs.io/en/latest/guide/ddd.html) and so ready to be applied by Mitiq users. This is still very new technique and so, if you notice any bugs, please let us know by
opening [issues](https://github.com/unitaryfoundation/mitiq/issues) on GihHub.
A further notable addition is the function [generate_quantum_volume_circuit()](https://mitiq.readthedocs.io/en/latest/apidoc.html#mitiq.benchmarks.quantum_volume_circuits.generate_quantum_volume_circuit) by @nickdgardner, extending the Mitiq benchmarking module with quantum volume
circuits.

Congratulations to the new member of the Mitiq team @natestemen and special thanks to the external contributors @Aaron-Robertson, @nickdgardner, @ZhaoyiLi-HekJukZaaiZyuJan and @amirebrahimi!

### All Changes

- Add section 2 (use case) on DDD to user guide (@nathanshammah, gh-1304)
- Update DDD options documentation (third section) (@andreamari, gh-1327)
- Add additional options subsection for DDD (third section) (@Aaron-Robertson, gh-1326)
- Add theory subsection for DDD (fifth section) (@andreamari, @natestemen, gh-1311)
- Update amazon-braket-sdk requirement from ~=1.19.0 to ~=1.24.0 (@dependabot gh-1324, gh-1310, gh-1306, gh-1303, gh-1297, gh-1278, gh-1271)
- Ensure lists in documentation are displayed properly (@natestemen, gh-1322)
- Add quantum volume circuits to benchmarking module (@nickdgardner, gh-1281)
- Update readme table with DDD (@andreamari, gh-1318)
- Test PR for unitaryhack test bounty (@nathanshammah, gh-1317)
- Fixes unexpected error from repeated rule (@Aaron-Robertson, gh-1316)
- Update mypy requirement from ~=0.950 to ~=0.960 (@dependabot, gh-1314)
- Use single URL when referring to google style (@natestemen, gh-1309)
- Update close-stale.yml (@andreamari, gh-1295)
- Update contributing docs (@amirebrahimi, gh-1305)
- Add fourth section of DDD docs (@andreamari, gh-1276)
- Fix citation file (@natestemen, gh-1299)
- Write the first sub-section of the DDD guide: How do I use DDD? (@Misty-W, gh-1277)
- Update qiskit requirement from ~=0.36.1 to ~=0.36.2 (@dependabot, gh-1302)
- Rename test_randomized_benchmaking.py to test_randomized_benchmarking.py (@nickdgardner, gh-1284)
- Fix broken links in docs (@andreamari, gh-1282)
- Create file structure (@andreamari, gh-1275)
- Update pennylane requirement from ~=0.23.0 to ~=0.23.1 (@dependabot, gh-1274)
- Fixed typo in documentation formula (@ZhaoyiLi-HekJukZaaiZyuJan, gh-1246)
- Update version to 0.16dev (@nathanshammah, gh-1268)

## Version 0.15.0 (April 29th, 2022)

### Summary

This milestone focused on updating dependencies and making progress on two new features, dynamical decoupling and learning based PEC. For dynamical decoupling, high-level functions and rules were added. For learning-based PEC, a function calculating representations with a biased (combination of depolarizing and dephasing) noise model was added. Several high priority bugs and issues were also fixed.

Special thanks to new contributors @RubidgeCarrie and @nickdgardner for their contributions to this release!

### All Changes

- Add digital dynamical decoupling high-level functions (@andreamari, gh-1251)
- Fix qubit naming in OperationRepresentations for Qiskit circuits (@Misty-W, gh-1238)
- Fix broken link in contributing guide (@Misty-W, gh-1257)
- Insert sequences for dynamical decoupling (@Aaron-Robertson, gh-1221)
- Short sequences in DDD API docs (@andreamari, gh-1237)
- Update pennylane-qiskit requirement from ~=0.22.0 to ~=0.23.0 (@dependabot, gh-1249)
- Update qiskit requirement from ~=0.36.0 to ~=0.36.1 (@dependabot, gh-1242)
- Update pennylane requirement from ~=0.22.2 to ~=0.23.0 (@dependabot, gh-1248)
- Move the gh pages deploy to a manually triggered action (@crazy4pi314, gh-1243)
- Update AUTHORS (@nickdgardner, gh-1239)
- fix typo (@nickdgardner, gh-1240)
- Add DDD to Api-Doc, fixes #1209 (@nathanshammah, gh-1226)
- Add rules for dynamical decoupling (@Aaron-Robertson, gh-1202)
- Fix compilation to native gateset in Braket notebook (@andreamari, gh-1225)
- Update amazon-braket-sdk requirement from ~=1.17.0 to ~=1.19.0 (@ dependabot, 1235)
- Function to represent OperationRepresentations for biased noise (@Misty-W, gh-1233)
- Bump codecov/codecov-action from 2.1.0 to 3.0.0 (@dependabot, gh-1216)
- Bump actions/stale from 4 to 5 (@dependabot, gh-1217)
- Update sphinxcontrib-bibtex requirement from ~=2.4.1 to ~=2.4.2 (@dependabot, gh-1218)
- Update cirq requirement from ~=0.14.0 to ~=0.14.1 @dependabot, gh-1219)
- Added citiation file for the repo (RubidgeCarrie, gh-1200)
- Add mitigate_executor and cdr_decorator to mitiq.cdr (@Aaron-Robertson, gh-1204)
- Update qiskit requirement from ~=0.35.0 to ~=0.36.0 (@dependabot, gh-1213)

## Version 0.14.0 (April 6th, 2022)

### Summary

This milestone focused on updating dependencies and making progress on two new features, dynamical decoupling and learning based error mitigation techniques. A number of high priority bugs and issues also were fixed.

### All Changes

- Update pennylane requirement from ~=0.22.1 to ~=0.22.2 (@dependabot, gh-1203)
- Update mypy requirement from ~=0.931 to ~=0.942 (@dependabot, gh-1183)
- Update pennylane-qiskit requirement from ~=0.20.0 to ~=0.22.0 (@dependabot, gh-1170)
- Update pennylane requirement from ~=0.21.0 to ~=0.22.1 (@dependabot, gh-1172)
- Update qiskit requirement from ~=0.34.2 to ~=0.35.0 (@dependabot, gh-1199)
- Update black requirement from ~=22.1 to ~=22.3 (@dependabot @andreamari @crazy4pi314, gh-1191)
- Avoid creation of empty moments when using fold_global (@andreamari, gh-1196)
- Merge pull request #1198 from unitaryfund/1140-raise-error-if-executo (@Misty-W)
- Port changes from old branch - workaround (@Misty-W)
- Merge pull request #1195 from unitaryfund/update-qiskit (@Misty-W)
- Update docs/source/guide/pec-3-options.myst @crazy4pi314 (@Misty-W)
- Merge branch 'master' into update-qiskit (@crazy4pi314)
- Update Cirq to 0.14 (@andreamari, gh-1193)
- fix pec-3-options notebook problem (@andreamari)
- Remove option to squash moments in global folding (@purva-thakre @andreamari, gh-1113)
- update qiskit and fix conversions (@andreamari)
- Idle qubits in conversion (@Aaron-Robertson, gh-1185)
- Add get_circuit_mask() for dynamical decoupling (@Aaron-Robertson, gh-1178)
- Update pydata-sphinx-theme requirement from ~=0.8.0 to ~=0.8.1 (@dependabot, gh-1188)
- Add get_slack_matrix_from_circuit_matrix() function for dynamical decoupling (@andreamari @Aaron-Robertson,gh-1178)
- Fix sphinx dependency problem (@andreamari , gh-1181)
- Create file structure for digital dynamical decoupling (@andreamari, gh-1175)
- adding pages publish step (@crazy4pi314,gh-1135)
- cast gate exponents to float (@andreamari,gh-1174)
- Update release.rst (@andreamari,gh-1145)
- Bump actions/setup-python from 2 to 3 (@dependabot, gh-1149)
- Bump actions/checkout from 2 to 3 (@dependabot, gh-1151)
- Update amazon-braket-sdk requirement from ~=1.15.0 to ~=1.17.0 (@dependabot, gh-1153)
- fixing install permissions, container user is root (@crazy4pi314 , gh-1147)

## Version 0.13.0 (February 25th, 2022)

### Summary

Mitiq is now compatible with the latest version (0.13.1) of Cirq! This update was blocked for a long time because of some technical difficulties. So, many thanks to @vtomole for finding a solution to this issue!
This should solve several dependency conflicts or warnings that you may have got when running `pip install mitiq` or `pip install -U mitiq`.

The HTML rendering of all PyQuil examples in our documentation is now fixed. Thanks @astrojuanlu for useful suggestions about readthedocs!

We also thank @Rahul-Mistri for adding GHZ circuits to our benchmarking module and for making Clifford circuits compatible with the Mitiq CDR technique (instead of raising an error as it happened before this release).

We discussed and approved the design documents (RFC) for two new error-mitigation techniques: _learning-based PEC_ and _digital dynamical decoupling_. You can find them at [this link](https://github.com/unitaryfoundation/mitiq/projects/7). Special thanks go to @Misty-W and @Aaron-Robertson!

### All Changes

- Add pre-executed pyquil notebooks (@andreamari, gh-1142)
- Fix optimal representation tests and unskip one of them (@andreamari gh-1141)
- Update amazon-braket-sdk requirement from ~=1.11.1 to ~=1.15.0 (@dependabot, gh-1137, gh-1116, gh-1108, gh-1105)
- Update black requirement from ~=19.10b0 to ~=22.1 (@dependabot, @crazy4pi314, gh-1110)
- Bump actions/github-script from 5 to 6 (@dependabot, gh-1129)
- Update mypy requirement from ~=0.930 to ~=0.931 (@dependabot, gh-1078)
- docs: add vtomole as a contributor for test, code (@allcontributors, @andreamari, gh-1132)
- docs: add Rahul-Mistri as a contributor for test, code (@allcontributors, @andreamari, gh-1130)
- docs: add L-P-B as a contributor for test, code (@allcontributors, gh-1131)
- Update PR template (@nathanshammah, gh-1117)
- Remove unused functions from `cirq_utils` and fix non-deterministic tests. (@andreamari gh-1123)
- Update pennylane requirement from ~=0.20.0 to ~=0.21.0 (@dependabot, gh-1122)
- Bump cirq version from 0.10.0 to 0.13.0 (@vtomole, gh-988)
- Can use Clifford Circuits with `execute_with_cdr` (@Rahul-Mistri, gh-1104)
- Docstring for GHZ-circuits reformatted (@Rahul-Mistri, gh-1101)

## Version 0.12.0 (January 21st, 2022)

### Summary

This release contains a considerable overhaul of the documentation organization and content:

- The guide is now divided into Core concepts and a new presentation of the quantum errror mitigation techniques (ZNE, PEC and CDR). Each technique contains subsections that explain with code snippets how to use them in Mitiq (gh-1021, gh-1004 gh-1031, gh-1099). Also the API doc has been extended and improved. Many thanks to @purvathakre @Misty-W for their help on rewriting the documentation and reviewing the pull requests.
- An example on how to use ZNE to improve the calculations of the energy potential landscape of molecular Hydrogen using VQE was added by @andreamari.

**New features**

- GHZ circuits were added to the benchmark subpackage by @Rahul-Mistri.
- Airspeed-velocity (asv) has been added to the CI by @rmlarose.

### All Changes

- Add core concepts guide page (@crazy4pi314, @nathanshammah, @andreamari, gh-1053)
- Add cdr-2-use-case.myst (gh-1099) (@andreamari, @nathanshammah)
- CDR documentation reorg (@nathanshammah, @andreamari, @crazy4pi314, gh-1031)
- Add molecular Hydrogen example (@andreamari, gh-1087)
- Update pydata-sphinx-theme requirement from ~=0.7.2 to ~=0.8.0 (@dependabot, gh-1091)
- Use raw.execute in asv benchmarks and remove GHZ circuits from asv (@rmlarose)
- Added GHZ circuits to benchmark (@Rahul-Mistri, gh-1089)
- Update scipy requirement from ~=1.7.2 to ~=1.7.3 (@dependabot
  , gh-1084)
- PR to trigger CI and fix a broken link (@andreamari, gh-1082)
- Fix readthedocs by changing a LaTex equation (@andreamari, @nathanshammah, gh-1077)
- Add asv benchmarking framework (@rmlarose, gh-1047)
- Update zne-3-options.myst (@andreamari, gh-1075)
- Update API doc with REM, executors and observables (@nathanshammah, gh-1050)
- ZNE Guide Reorg (@crazy4pi314, @Misty-W, @purva-thakre, @nathanshammah, @andreamari, gh-1021)
- Fix broken link on master. (@andreamari, gh-1069)
- Update mypy requirement from ~=0.910 to ~=0.930 (@dependabot, gh-1065)
- Docs: add AkashNarayanan as a contributor for infra (@allcontributors, gh-1059)
- Fixes #1034 release docs update (@crazy4pi314, @nathanshammah, gh-1054)
- Docs: add DSamuel1 as a contributor for code (@allcontributors, @Misty-W, gh-1056)
- Docs: add Misty-W as a contributor for code, example (@allcontributors, gh-1055)
- Docs: update README.md (@allcontributors)
- Update PEC docs (@andreamari, @nathanshammah, @rmlarose, gh-1004)
- Revert "Update mypy requirement from ~=0.910 to ~=0.920 (@nathanshammah, gh-1052)
- Update mypy requirement from ~=0.910 to ~=0.920 (@dependabot, gh-1051)
- Update pytest-xdist[psutil] requirement from ~=2.4.0 to ~=2.5.0 (@dependabot, gh-1044)
- Update amazon-braket-sdk requirement from ~=1.11.0 to ~=1.11.1 (@dependabot, gh-1042)
- Update pennylane requirement from ~=0.19.0 to ~=0.19.1 (@dependabot, gh-1029)
- Update amazon-braket-sdk requirement from ~=1.9.5 to ~=1.11.0 (@dependabot, gh-1038)

## Version 0.11.1 (November 29th, 2021)

### Summary

This patch release fixes two bugs:

- Bug: PEC could only be used with `cirq.Circuit`s, not `mitiq.QPROGRAM`, due to a missing conversion.
  - Fix: PEC can now be used with any `mitiq.QPROGRAM` (gh-1018).
- Bug: CDR classically simulated the wrong circuits when doing regression.
  - Fix: The correct circuits are now classically simulated (gh-1026).

Also fixes a smaller bug where some tools in `mitiq.interface.mitiq_qiskit` modified `qiskit.QuantumCircuit`s when they shouldn't.

### All Changes

- Update scipy requirement from ~=1.7.1 to ~=1.7.2 (@dependabot, gh-1017)
- CDR: Run the training circuits on the simulator (@rmlarose and @andreamari, gh-1026).
- Update scipy requirement from ~=1.7.1 to ~=1.7.2 (@dependabot, gh-1017)
- Update pydata-sphinx-theme requirement from ~=0.7.1 to ~=0.7.2 (@dependabot, gh-1024)
- Update qiskit requirement from ~=0.31.0 to ~=0.32.0 (@dependabot, gh-1025)
- Update pydata-sphinx-theme requirement from ~=0.7.1 to ~=0.7.2 (@dependabot, gh-1024)
- [Bug fix] Avoid circuit mutation in qiskit executors (@andreamari, gh-1019)
- [Bug fix] Add back-conversions in execute_with_pec (@andreamari, gh-1018)
- Increase shots in zne tests with shot_list (@andreamari, gh-1020)
- Update pennylane requirement from ~=0.18.0 to ~=0.19.0 (@nathanshammah, gh-1022)
- Add workflow figures and technique descriptions (@nathanshammah, gh-953)
- Prepare to release 0.11.0 (@rmlarose, gh-1010)

## Version 0.11.0 (November 3rd, 2021)

### Summary

This release introduces `Observable`s as a major new feature to the Mitiq workflow and a few breaking changes. Support
for Pennylane has been added by adding `pennylane.QuantumTape`s to `mitiq.QPROGRAM`.

**New features**

- Specify and use a `mitiq.Observable` in any error-mitigation technique.

  - This means the `executor` function does not have to return the expectation value as a `float` anymore, but rather
    can return a `mitiq.QuantumResult` - i.e., an object from which the expectation value can be computed provided
    an observable.
  - The `executor` function can still return a `float`, in which case the `Observable` does not need to be specified
    (and should not be specified).

- All error mitigation techniques can now use batching with the same interface.

- PEC can be run with only a subset of representations of the gates in a circuit. In other words, if the circuit has
  two gates, `H` and `CNOT`, you can run `execute_with_pec` by only providing an `OperationRepresentation` for, e.g.,
  the `CNOT`.

  - Before, you had to provide all representations or an error would be raised.
  - Performance of PEC may be better by providing all `OperationRepresentation`s. This change is only with respect to
    usability and not performance.

- Circuits written in `Pennylane` (as `QuantumTape`s) are now recognized and suppported by Mitiq.

**Breaking changes**

- Signatures of `execute_with_xxx` error-mitigation techniques have changed to include `Observable`s. The default value
  is `None`, meaning that the `executor` should return a `float` as in the old usage, but the additional argument and
  change to keyword-only arguments (see below) may require you to make updates.

- You must now use provide keywords for technique-specific arguments in error mitigation techniques. Example:

```python
# Example new usage of providing keyword arguments. Do this.
execute_with_pec(circuit, executor, observable, representations=representations)
```

instead of

```python
# Old usage. Don't do this. Technique-specific arguments like `representations` are now keyword-only.
execute_with_pec(circuit, executor, observable, representations)
```

The latter will raise `# TypeError: execute_with_pec() missing 1 required keyword-only argument: 'representations'`.

- The first argument of `execute_with_zne` is now `circuit` instead of `qp` to match signatures of other
  `execute_with_xxx` functions.

### All Changes

- Increase shots in test_zne.py (@andreamari, gh-1011)
- Refactor `qp` to `circuit` in `execute_with_zne` (@rmlarose, gh-1009)
- Add `Observable` documentation and `Observable.from_pauli_string_collections` method (@rmlarose, gh-1007)
- Executor docs (@rmlarose, gh-1008)
- Bump qiskit to version 0.31 and pin it explicitly in dev requirements (@andreamari, gh-993)
- Use `Executor.evaluate` for batching in ZNE (@rmlarose, gh-1005)
- Add `Executor.evaluate` and use for batched execution (@rmlarose, gh-1001)
- Bump PyQuil (@rmlarose, gh-992)
- CDR with Observables (@rmlarose, gh-985)
- Fix bug in OperationRepresentation printing (@andreamari, gh-975)
- Ignore patch releases in dependabot (@rmlarose, gh-981)
- Update pydata-sphinx-theme requirement from ~=0.6.3 to ~=0.7.1 (@dependabot, gh-962)
- Update flake8 requirement from ~=3.9.2 to ~=4.0.1 (@dependabot, gh-982)
- Add PennyLane to frontend table in the readme (@andreamari, gh-973)
- Update amazon-braket-sdk requirement from ~=1.9.1 to ~=1.9.5 (@dependabot, gh-970)
- Update pytest-cov requirement from ~=2.12.1 to ~=3.0.0 (@dependabot, gh-963)
- Fix pip package resolving problems (@andreamari, gh-976)
- Add support for pennylane circuits (everybody and their grandmother, gh-836)
- Keyword only arguments in execute_with_technique functions (@rmlarose, gh-971)
- Foldability check includes a check for inverse (@purva-thakre, gh-939)
- Bump actions/github-script from 3 to 5 (@dependabot, gh-969)
- PEC with Observables & skip operations without known representations (@rmlarose, gh-954)
- Update qiskit-terra requirement from ~=0.18.2 to ~=0.18.3 (@dependabot, gh-955)
- Add observable to zne_decorator (@rmlarose, gh-967)
- Update qiskit-ibmq-provider requirement from ~=0.16.0 to ~=0.17.0 (@dependabot, gh-965)
- Binder badge workflow (@AkashNarayanan, gh-964)
- Fix links and typos in vqe-pyquil-demo.myst and pyquil_demo.myst (@Misty-W, gh-959)
- ZNE with Observables (@rmlarose, gh-948)
- Add PauliString multiplication (@rmlarose, gh-949)
- Update amazon-braket-sdk requirement from ~=1.9.0 to ~=1.9.1 (@dependabot, gh-950)
- Fixing changelog and improving release documentation (@crazy4pi314, gh-946)
- Update version to 0.11.0dev (@rmlarose, gh-947)
- Update pytest-xdist[psutil] requirement from ~=2.3.0 to ~=2.4.0 (@dependabot, gh-938)
- Fix AWS example by fixing a bug in mirror circuits (@andreamari, gh-940)
- Bump codecov/codecov-action from 2.0.3 to 2.1.0 (@dependabot, gh-922)
- Improve OperationRepresentation printing (@andreamari, gh-901)
- Updating release process (@crazy4pi314, gh-936)

Huge thanks to all contributors on this release! @Misty-W, @AkashNarayanan, @purva-thakre, @trbromley, @nathanshammah,
@crazy4pi314, @andreamari, and @rmlarose.

## Version 0.10.0 (September 17, 2021)

### Summary

This release adds a ton of new stuff, both error mitigation tools as well as infrastructure upgrades.
Some highlights:

- New integration with [AWS Braket](https://aws.amazon.com/braket/)
- A new pyQuil example by @Misty-W and lots of pyQuil debugging.
- Support for mirror circuits by @DSamuel1.
- Dependabot is now helping us keep our dependencies up to date.
- Lots of documentation fixes and features like a gallery view of the examples and mybinder.org support.
- New `Observable` and `MeasurementResult` dataclass.

Thanks to @Misty-W and @DSamuel1 for their great contributions this release! 🎉

### All Changes

- Reduce noise in braket example (@andreamari, gh-933)
- Add ZNE example on AWS Braket (@rmlarose, gh-929)
- A few changes / fixes to mirror circuits (@rmlarose, gh-928)
- Change Binder link from jupyterlab to classic view (@andreamari, gh-925)
- Dependabot settings patch: one dependancy per line (@rmlarose, gh-921)
- Dependabot settings (@rmlarose, gh-920)
- Update sphinxcontrib-bibtex requirement from ~=2.3.0 to ~=2.4.1 (@dependabot, gh-919)
- Rename Mitiq Examples -> Examples (@nathanshammah, gh-916)
- Update qiskit-terra requirement from ~=0.18.1 to ~=0.18.2 (@dependabot, gh-912)
- Add `Observable.expectation` with support for measurement results and density matrices; Refactor `Collector` -> `Executor` (@rmlarose, gh-904)
- Install mitiq in readthedocs to avoid mitiq.py files (@andreamari, gh-903)
- Crazy4pi314/example gallery (@crazy4pi314, gh-902)
- Import mirror circuits (@rmlarose, gh-900)
- Mirror Circuits Update: Resolves #890 and #891 (@DSamuel1, gh-895)
- Fix html rendering problems of the PEC example (@andreamari, gh-894)
- Add `qubit_indices` to `MeasurementResult` (@rmlarose, gh-892)
- Update mirror circuits docstrings (@rmlarose, gh-889)
- Update flake8 requirement from ~=3.7.9 to ~=3.9.2 (@dependabot, gh-887)
- Update pytest-cov requirement from ~=2.11.1 to ~=2.12.1 (@dependabot, gh-885)
- Update sphinx-copybutton requirement from ~=0.3.0 to ~=0.4.0 (@dependabot, gh-884)
- Update mypy requirement from ~=0.812 to ~=0.910 (@dependabot, gh-883)
- Update pytest-xdist[psutil] requirement from ~=2.2.1 to ~=2.3.0 (@dependabot, gh-880)
- Update sphinxcontrib-bibtex requirement from ~=2.2.0 to ~=2.3.0 (@dependabot, gh-879)
- Update amazon-braket-sdk requirement from ~=1.5.10 to ~=1.8.0 (@dependabot, gh-878)
- Bump codecov/codecov-action from 1.3.1 to 2.0.3 (@dependabot, gh-875)
- Bump actions/stale from 3.0.19 to 4 (@dependabot, gh-874)
- Updating pinned Scipy version (@crazy4pi314, gh-871)
- Add some missing package metadata causing problems (@crazy4pi314, gh-870)
- Fixing syntax for dependabot to make it work correctly (@crazy4pi314, gh-866)
- Make `MeasurementResult` a dataclass; Add expectation from measurements (@rmlarose, gh-860)
- Mirror circuits (@DSamuel1, gh-859)
- Fix link in README (@andreamari, gh-856)
- Add step-by-sep tutorial on PEC (@andreamari, gh-854)
- Adding mitiq survey to readme (@crazy4pi314, gh-853)
- Add `Observable` (@rmlarose, gh-852)
- Manual PyPI deployment trigger (@crazy4pi314, gh-851)
- Make sure stale marking of issues happens (@crazy4pi314, gh-849)
- Fix invisible output in mitiq codeblocks (@andreamari, gh-847)
- update references in README (@andreamari, gh-846)
- Update to latest Qiskit (@rmlarose, gh-845)
- Set measurement result type and add post-selection (@rmlarose, gh-844)
- pyQuil parametric compilation example (@Misty-W, gh-843)
- Add My Binder (@nathanshammah, gh-841)
- Show output of Mitiq paper code blocks in example on RTD (@nathanshammah, gh-840)
- Add code blocks from the paper (@nathanshammah, gh-838)
- Make ZNE (more) usable with PyQuil (@rmlarose, gh-835)
- Improve PEC efficiency with batched sampling (@andreamari, gh-833)
- Fix warnings in docs log output during build (@andreamari, gh-832)
- Update readme and remove research.rst (@rmlarose, gh-831)
- Add linkcheck to docs build and fix broken links (@rmlarose, gh-827)
- Initial support for executors which return measurement results: part 1/2 (@rmlarose, gh-826)
- Adding table of techniques to readme (@crazy4pi314, gh-825)
- Move example/template notebook to contributing TOC (@rmlarose, gh-814)
- Fix multiplication order when adding NoisyOperations (@andreamari, gh-811)
- Better error message for `CircuitConversionError`s (@rmlarose, gh-809)
- Fix some documentation not being tested & remove global imports in docs config (@rmlarose, gh-804)

## Version 0.9.3 (July 7th, 2021)

### Summary

This primary reason for this patch release is to fix a bug interfacing with Qiskit circuits (gh-802).

### All Changes

- [Docs] Add CDR to README and braket to Overview (@rmlarose, gh-778).
- Rename parameter calibration function and make it visible (@rmlarose, gh-780).
- Allow adding qubits when transforming registers in a Qiskit circuit (@rmlarose, gh-803).

## Version 0.9.2 (June 30th, 2021)

### Summary

This patch release fixes a Braket integration bug (gh-767).
It also adds an example about Clifford data regression in the documentation.

### All Changes

- Ensure short circuit warning is multi-platform (@andreamari gh-769).
- Add CDR example to docs + small change to `cdr.calculate_observable` (@rmlarose, gh-750).

## Version 0.9.1 (June 24th, 2021)

### Summary

This is a patch release to fix two bugs (gh-736, gh-737) related to the integration with optional packages.
It also fixes other minor problems (see the list of changes below).

### All Changes

- Patch 0.9.0 (@rmlarose, gh-739).
- Make readthedocs succeed in building the pdf with pdflatex (@andreamari, gh-743).
- Update energy landscape example in docs (@andreamari, gh-742).
- Remove old deprecation warnings (@rmlarose, gh-744).

## Version 0.9.0 (June 17th, 2021)

### Summary

The main addition introduced in this release is the implementation of a new error mitigation technique: (variable-noise) Clifford data regression ([arXiv:2005.10189](https://arxiv.org/abs/2005.10189), [arXiv:2011.01157](https://arxiv.org/abs/2011.01157)). This is structured as
a Mitiq module called `mitiq.cdr`.

Another important change is the integration with Amazon Braket, such that Mitiq is now compatible with circuits of type `braket.circuits.Circuit`. Moreover all the existing Mitiq integrations are now organized into unique module called `mitiq.interface`.

In the context of probabilistic error cancellation, the sub-module `mitiq.pec.representations` has been significantly improved. Now one can easily compute the optimal quasi-probabiliy representation of an ideal gate as a linear combination of `NoisyOperation` objects.

Thanks to all contributors (@L-P-B, @Aaron-Robertson, @francespoblete, @LaurentAjdnik, @maxtremblay, @andre-a-alves, @paniash, @purva-thakre) and in particular to the participants of [unitaryHACK](https://2021.unitaryhack.dev/)!

### All Changes

- New notation NoisyOperation(circuit, channel_matrix) (@andreamari, gh-725).
- Update docs for transforming Qiskit registers (@rmlarose, gh-724).
- Remove classical register transformations in Qiskit conversions (@Aaron-Robertson, gh-672).
- Change return format for `execute_with_cdr` (@rmlarose, gh-722).
- Organize supported packages in `mitiq.interface` (@rmlarose, gh-706).
- Change requirements to Cirq 0.10 (@andreamari, gh-717).
- Make CDR work with any `QPROGRAM` (@rmlarose, gh-718).
- Clearer return type for extrapolate and remove deprecated method (@rmlarose, gh-714).
- Update feature request template (@rmlarose, gh-713).
- CDR part two - Clifford data regression functions to fit training data (@L-P-B gh-677).
- Consolidate images, update README. (@rmlarose, gh-709).
- New logo for readme (@crazy4pi314, @francespoblete, gh-709).
- Update Guidelines for Release (@nathanshammah, gh-707).
- Add action to close stale issues/pr #698 (@crazy4pi314, gh-699).
- Optimal quasi-probability representations (@andreamari, gh-701).
- Update links to Cirq documentation (@LaurentAjdnik, gh-704).
- Warning for short programs [unitaryHACK] (@Yash-10, gh-700).
- Preparing for optimal representations: Helper functions for channel manipulation. (@andreamari, gh-694).
- [UnitaryHACK] Improve conversion from braket to cirq (@maxtremblay, gh-688).
- [unitaryHack] Add instructions to solve make docs error for Windows/3.8 users (@andre-a-alves, gh-691).
- Add link to docs for source installation in README (@paniash, gh-682).
- Mention Braket in the README (@andreamari, gh-687).
- [UnitaryHACK] ZNE-PEC uniformity (@andre-a-alves, gh-656).
- Minor update of parameter noise scaling (@andreamari, gh-684).
- Add braket support via rudimentary translator (@rmlarose, gh-590).
- Specify qiskit elements in output of about.py (@purva-thakre, gh-674).
- Depend only on cirq-core (@rmlarose, gh-673). This was reverted in gh-717.
- Fix docs build (@rmlarose, gh-671).
- Fix is_clifford logic for Clifford Data Regression (@L-P-B gh-669).

## Version 0.8.0 (May 6th, 2021)

### Summary

This release has the following major components:

- Re-implements local folding functions (`zne.scaling.fold_gates_from_left`, `zne.scaling.fold_gates_from_right` and `zne.scaling.fold_gates_at_random`) to make them more uniform at large scale factors and match how they were defined in [Digital zero-noise extrapolation for quantum error mitigation](https://arxiv.org/abs/2005.10921).
- Adds a new noise scaling method, `zne.scaling.fold_all`, which locally folds all gates. This can be used, e.g., to "square CNOTs," a common literature technique, by calling `fold_all` and excluding single-qubit gates.
- Adds functionality for the training portion of [Clifford data regression](https://arxiv.org/abs/2005.10189), specifically to map an input (pre-compiled) circuit to a set of (near) Clifford circuits which are used as training data for the method. The full CDR technique is still in development and will be complete with the addition of regression methods.
- Improves the (sampling) performance of PEC (by a lot!) via fewer circuit conversions.
- Adds `PauliString` object, the first change of several in the generalization of executors. This object is not yet used in any error mitigation pipelines but can be used as a stand-alone.

Additionally, this release

- Fixes some CI components including uploading coverage from master and suppressing nightly Test PyPI uploads on forks.
- Adds links to GitHub on README and RTD.

Special thanks to all contributors - @purva-thakre, @Aaron-Robertson, @andre-a-alves, @mstechly, @ckissane, @HaoTy, @briancylui, and @L-P-B - for your work on this release!

### All Changes

- Remove docs/pdf in favor of RTD (@rmlarose, gh-662).
- Redefine and re-implement local folding functions (@andreamari, gh-649).
- Move Cirq executors from docs to utilities (@purva-thakre, gh-603).
- Add functionality for the training portion of Clifford Data Regression (@L-P-B, gh-601).
- Fix custom factory example in docs (@andreamari, gh-601).
- Improves PEC sampling speed via fewer conversions (@ckissane, @HaoTy, @briancylui, gh-647).
- Minor typing fixes (@mstechly, gh-652).
- Add new `fold_all` scaling method (@rmlarose, gh-648).
- Add `PauliString` (@rmlarose, gh-633).
- Update information about formatting in RTD (@purva-thakre, gh-622).
- Fix typo in getting started guide (@andre-a-alves, gh-640).
- Suppress Test PyPI nightly upload on forks (@purva-thakre, gh-597).
- Add link to GitHub in README and add octocat link to GitHub on RTD (@andre-a-alves, gh-637).
- Move all benchmark circuit generation to `mitiq.benchmarks`, adding option for converions (@Aaron-Robertson, gh-632).
- Use `execute_with_shots_and_noise` in Qiskit utils test (@Aaron-Robertson, gh-621).
- Install only the Qiskit packages we need (@purva-thakre, gh-614).
- Update PR template (@rmlarose, gh-634).
- Add blurb about unitaryHACK (@nathanshammah).

## Version 0.7.0 (April 7th, 2021)

### Summary

This release focuses on contributions from the community.
Many thanks @yhindi for adding a method to parametrically scale noise in circuit, to @aaron-robertson and @pchung39 for adding Qiskit executors with depolarizing noise to the utils,
to @purva-thakre for several bug fixes and improvements. Thanks also to @BobinMathew and @marwahaha for typo corrections.

### All Changes

- Pin `docutils` version to solve CI issues (@purva-thakre, gh-626)
- Add qiskit executor example for exact density matrix simulation with depolarizing noise (@aaron-robertson, gh-574)
- Replace Qiskit utils with Qiskit executors from docs (@pchung39, @aaron-robertson, gh-584)
- Change custom depolarizing channel in PEC (@purva-thakre, gh-615)
- Document codecov fetch depth change and increase codecov action version (@grmlarose, gh-618)
- Fix spelling typo (@marwahaha, gh-616)
- Add link for MEP (@purva-thakre, gh-610)
- Correct a typo in the readme (@BobinMathew, gh-609)
- Add `represent_operations_in_circuit_...` functions for PEC (@andreamari, gh-515)
- Add qiskit and cirq executor examples gifs to readme (@nathanshammah, gh-587)
- Move Cirq executors in docs to cirq_utils (@purva-thakre, gh-603)
- Add parametrized scaling (@yhindyYousef, gh-411)
- Fix mitiq.about() qiskit version (@aaron-robertson, gh-598)
- Fix typo in ZNE documentation (@purva-thakre, gh-602)
- Add minimal section on PEC to documentation (@nathanshammah @andreamari, gh-564)
- Various improvements to CI and testing (@rmlarose, gh-583)
- Corrects typo in docs of error mitigation guide (@purva-thakre, gh-585)
- Remove factory equality and fix bug in `PolyExpFactory.extrapolate` (@rmlarose, gh-580)
- [Bug Fix] Examples (notebooks) in the online documentation now have output code cells (@Aaron-Robertson, gh-576)
- Update version to dev (@andreamari, gh-572)
- Change custom depolarizing channel in PEC test (@purva-thakre, gh-615)

## Version 0.6.0 (March 1st, 2021)

### Summary

The automated workflows for builds and releases are improved and PyPI releases are now automated.
We have more documentation on PEC and have a new tutorial on QAOA with Mitiq, as well as some miscellaneous bug fixes.

### All Changes

- Add minimal section on PEC to documentation (@nathanshammah, gh-564)

- Improve CI and release/patch workflow and documentation (@crazy4pi314 gh-566).
- Adding a Mitiq Enhancement Proposal template and process (@crazy4pi314 gh-563).
- Add to the documentation papers citing Mitiq, close gh-424 (@nathanshammah gh-560).
- Added a new tutorial where MaxCut is solved with a mitigated QAOA (@andreamari, gh-562).
- Bumping the version of Qiskit version in `dev_requirements` (@andreamari gh-554)
- Retain measurement order when folding Qiskit circuits (@rmlarose gh-557)
- Standardize and touch up docs (@rmlarose gh-553)
- Adding make doc-clean command (@crazy4pi314 gh-549)
- Fix and refactor RB circuits function (@rmlarose gh-539)
- Improve GateOperation simplification in exponents (@rmlarose gh-541)

## Version 0.5.0 (February 8th, 2021)

### Summary

The implementation of Probabilistic Error Cancellation is now multi-platform and
documented in the docs (for the moment only in the _Getting Started_ section).
A new infrastructure based on [MyST](https://myst-parser.readthedocs.io/en/stable/)
can now be used for writing the documentation and, in particular, for adding new examples
in the _Mitiq Examples_ section.

### All Changes

- Adding documentation section for examples, support for Jyupyter notebooks (@crazy4pi314, gh-509).
- Add minimal documentation for PEC in the Getting Started (@andreamari, gh-532).
- Optionally return a dictionary with all pec data in execute_with_pec (@andreamari, gh-518).
- Added local_depolarizing_representations and Choi-based tests (@andreamari, gh-502).
- Add multi-platform support for PEC (@rmlarose, gh-500).
- Added new `plot_data` and `plot_fit` methods for factories(@crazy4pi314 @elmandouh, gh-333).
- Fixes random failure of PEC sampling test(@rmlarose, gh-481).
- Exact copying of shell commands, update make target for pdf and update development version(@rmlarose, gh-469).
- Add a new FakeNodesFactory class based on an alternative interpolation method (@elmandouh, gh-444).
- Add parameter calibration method to find base noise for parameter noise (@yhindy, gh-411).
- Remove duplication of the reduce method in every (non-adaptive) factory (@elmandouh, gh-470).

## Version 0.4.1 (January 12th, 2021)

### Summary

This release fixes a bug in the docs.

### All Changes

- [Bug Fix] Ensure code is tested in IBMQ guide and adds a doctest target in Makefile(@rmlarose, gh-488)

## Version 0.4.0 (December 6th, 2020)

### Summary

This release adds new getter methods for fit errors, extrapolation curves, etc. in ZNE factory objects as well as
custom types for noisy operations, noisy bases, and decompositions in PEC. It also includes small updates and fixes
to the documentation, seeding options for PEC sampling functions, and bug fixes for a few non-deterministic test failures.

### All Changes

- Add reference to review paper in docs (@willzeng, gh-423).
- Add unitary folding API to RTD (@rmlarose, gh-429).
- Add theory subsection on PEC in docs (@elmandouh, gh-428).
- Fix small typo in documentation function name (@nathanshammah, gh-435).
- Seed Qiskit simulator to fix non-deterministic test failure (@rmlarose, gh-425).
- Fix formatting typo and include hyperlinks to documentation objects (@nathanshammah, gh-438).
- Remove error in docs testing without tensorflow (@nathanshammah, gh-439).
- Add seed to PEC functions (@rmlarose, gh-432).
- Consolidate functions to generate randomized benchmarking circuits in different platforms, and clean up pyquil utils (@rmlarose, gh-426).
- Add new get methods (for fit errors, extrapolation curve, etc.) to Factory objects (@crazy4pi314, @andreamari, gh-403).
- Update notebook version in requirements to resolve vulnerability found by security bot.(@nathanshammah, gh-445).
- Add brief description of noise and error mitigtation to readme (@rmlarose, gh-422).
- Fix broken links in documentation (@purva-thakre, gh-448).
- Link to stable RTD instead of latest RTD in readme (@rmlarose, gh-449).
- Add option to automatically deduce the number of samples in PEC (@andreamari, gh-451).
- Fix PEC sampling bug (@rmlarose, gh-453).
- Add types for PEC (@rmlarose, gh-408).
- Add warning for large samples in PEC (@sid1993, gh-459).
- Seed a PEC test to avoid non-deterministic failure (@andreamari, gh-460).
- Update contributing docs (@purva-thakre, gh-465).

## Version 0.3.0 (October 30th, 2020)

### Summary

Factories now support "batched" executors, meaning that when a backend allows
for the batch execution of a collection of quantum circuits, factories can now
leverage that functionality. In addition, the main focus of this release was
implementing probabilistic error cancellation (PEC), which was introduced in
[Temme2017][temme2017] as a method for quantum error mitigation. We completed
a first draft of the major components in the PEC workflow, and in the next
release plan to demonstrate the full end-to-end operation of the new technique.

[temme2017]: https://arxiv.org/abs/1612.02058

### All Changes

- Fix broken links on the website (@erkska, gh-400).
- Use cirq v0.9.0 instead of cirq-unstable (@karalekas, gh-402).
- Update mitiq.about() (@rmlarose, gh-399).
- Refresh the release process documentation (@karalekas, gh-392).
- Redesign factories, batch runs in BatchedFactory, fix Qiskit utils tests (@rmlarose, @andreamari, gh-381).
- Add note on batched executors to docs (@rmlarose, gh-405).
- Added Tensorflow Quantum executor to docs (@k-m-schultz, gh-348).
- Fix a collection of small build & docs issues (@karalekas, gh-410).
- Add optimal QPR decomposition for depolarizing noise (@karalekas, gh-371).
- Add PEC basic implementation assuming a decomposition dictionary is given (@andreamari, gh-373).
- Make tensorflow requirements optional for docs (@karalekas, gh-417).

Thanks to @erkska and @k-m-schultz for their contributions to this release!

## Version 0.2.0 (October 4th, 2020)

### Announcements

The preprint for Mitiq is live on the arXiv [here][arxiv]!

### Summary

This release centered on source code reorganization and documentation, as well
as wrapping up some holdovers from the initial public release. In addition, the
team began investigating probabilistic error cancellation (PEC), which will be the
main focus of the following release.

### All Changes

- Re-organize scaling code into its own module (@rmlarose, gh-338).
- Add BibTeX snippet for [arXiv citation][arxiv] (@karalekas, gh-351).
- Fix broken links in PR template (@rmlarose, gh-359).
- Add limitations of ZNE section to docs (@rmlarose, gh-361).
- Add static extrapolate method to all factories (@andreamari, gh-352).
- Removes barriers in conversions from a Qiskit circuit (@rmlarose, gh-362).
- Add arXiv badge to readme header (@nathanshammah, gh-376).
- Add note on shot list in factory docs (@rmlarose, gh-375).
- Spruce up the README a bit (@karalekas, gh-383).
- Make mypy checking stricter (@karalekas, gh-380).
- Add pyQuil executor examples and benchmarking circuits (@karalekas, gh-339).

[arxiv]: https://arxiv.org/abs/2009.04417

## Version 0.1.0 (September 2nd, 2020)

### Summary

This marks the first public release of Mitiq on a stable version.

### All Changes

- Add static extrapolate method to all factories (@andreamari, gh-352).
- Add the `angle` module for parameter noise scaling (@yhindy, gh-288).
- Add to the documentation instructions for maintainers to make a new release (@nathanshammah, gh-332).
- Add basic compilation facilities, don't relabel qubits (@karalekas, gh-324).
- Update readme (@rmlarose, gh-330).
- Add mypy type checking to CI, resolve existing issues (@karalekas, gh-326).
- Add readthedocs badge to readme (@nathanshammah, gh-329).
- Add change log as markdown file (@nathanshammah, gh-328).
- Add documentation on mitigating the energy landscape for QAOA MaxCut on two qubits (@rmlarose, gh-241).
- Simplify inverse gates before conversion to QASM (@andreamari, gh-283).
- Restructure library with `zne/` subpackage, modules renaming (@nathanshammah, gh-298).
- [Bug Fix] Fix minor problems in executors documentation (@andreamari, gh-292).
- Add better link to docs and more detailed features (@andreamari, gh-306).
- [Bug Fix] Fix links and author list in README (@willzeng, gh-302).
- Add new sections and more explanatory titles to the documentation's guide (@nathanshammah, gh-285).
- Store optimal parameters after calling reduce in factories (@rmlarose, gh-318).
- Run CI on all commits in PRs to master and close #316 (@karalekas, gh-325).
- Add Unitary Fund logo to the documentation html and close #297 (@nathanshammah, gh-323).
- Add circuit conversion error + tests (@rmlarose, gh-321).
- Make test file names unique (@rmlarose, gh-319).
- Update package version from v. 0.1a2, released, to 0.10dev (@nathanshammah, gh-314).

## Version 0.1a2 (August 17th, 2020)

- **Initial public release**: on [Github][Github] and [PyPI][PyPI].

## Version 0.1a1 (June 5th, 2020)

- **Initial release (internal).**

[Github]: https://github.com/unitaryfoundation/mitiq
[PyPI]: https://pypi.org/project/mitiq/0.1a2/
