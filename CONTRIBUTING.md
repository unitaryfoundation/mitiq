# Contributing to Mitiq

All contributions to this project are welcome, and they are greatly appreciated; every little bit helps.
The most common ways to contribute here are

1. opening an [issue](https://github.com/unitaryfoundation/mitiq/issues/new) to report a bug or propose a new feature, or ask a question, and
2. opening a [pull request](https://github.com/unitaryfoundation/mitiq/pulls) to fix a bug, or implement a desired feature.
3. opening a [discussion post](https://github.com/unitaryfoundation/mitiq/discussions) to ask a question (all questions welcome!), provide feedback, or show something off!

The rest of this document describes the technical details of getting set up to develop, and make your first contribution to Mitiq.


## Development environment

Mitiq uses [uv](https://docs.astral.sh/uv/) for packaging and dependency management. It may also be used for managing the virtual environment. 

1. Ensure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the [Mitiq repository](https://github.com/unitaryfoundation/mitiq). If you are unfamiliar with `git`, check out the [docs](https://git-scm.com/), and learn about what the typical [`git` workflow](https://www.asmeurer.com/git-workflow/) looks like.
3. Inside the Mitiq directory (`cd mitiq`), use uv to create a virtual environment (/.venv) and install the dependencies by calling `make install`.

```bash
pip install uv
git clone https://github.com/unitaryfoundation/mitiq.git
cd mitiq
make install
```

You should now have a development environment set up to work on Mitiq! 🎉 To go forward with making the desired changes, please consult the ["Making changes" section](https://www.asmeurer.com/git-workflow/#making-changes) of the `git` workflow article. If you've encountered any problems thus far, please let us know by opening an issue! More information about workflow can be found below in the [lifecycle](#lifecycle) section.

```{note}

Since uv uses a virtual environment, any commands to be run inside the virtual environment will need to be prefaced with `uv run` or the uv [managed virtual environment](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment) will need to be activated by running `source .venv/bin/activate` in your shell. 

Running `make` commands do NOT require prepending `uv run`.
```


What follows are recommendations/requirements to keep in mind while contributing.

## Making changes

### Adding tests

When modifying and/or adding new code it is important to ensure the changes are covered by tests.
Test thoroughly, but not excessively.
Mitiq uses a nested structure for packaging tests in directories named `tests` at the same level of each module.
The only exception to this is that any tests requiring a QVM should be placed in the `mitiq_pyquil/tests` folder.

### Running tests

After making changes, ensure your changes pass all the existing tests (and any new tests you've added).
Use `uv run pytest mitiq/$MODULE` to run the tests for the module you are working on.
Once they pass, you can run the entire test suite (excluding those that require the pyQuil QVM) by running the following command.

```bash
make test
```

This can often be slow, however, so testing your changes [iteratively](https://docs.pytest.org/en/7.1.x/how-to/usage.html#specifying-which-tests-to-run) using `pytest` is often faster when doing development. 

To run the tests for the pyQuil plugins, run
```bash
make test-pyquil
```

To run all tests, run
```bash
make test-all
```

**Note**: For the pyQuil tests to run, you will need to have QVM & quilc servers
running in the background. The easiest way to do this is with [Docker](https://www.docker.com/) via

```bash
docker run --rm -idt -p 5000:5000 rigetti/qvm -S
docker run --rm -idt -p 5555:5555 rigetti/quilc -R
```

### Updating the documentation
Follow these [instructions for contributing to the documentation](contributing_docs.md) which include guidelines about updating the API-doc, adding examples, and updating the user guide.

### Style guidelines

Mitiq code is developed according the best practices of Python development.
- Please get familiar with [PEP 8](https://peps.python.org/pep-0008/) (code) and [PEP 257](https://peps.python.org/pep-0257/) (docstrings) guidelines.
- Use annotations for type hints in the objects' signature.
- Write [google-style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).

We use [Ruff](https://docs.astral.sh/ruff/) to automatically lint the code and enforce style requirements as part of the CI pipeline.
You can run these style tests yourself locally in the top-level directory of the repository.

You can check for linting/formatting violations with
```bash
make check-format
```
Many common issues can be fixed automatically using the following command, but some will require manual intervention to appease Ruff.
```bash
make format
```

We also use [Mypy](https://mypy.readthedocs.io/en/stable/) as a type checker to find incompatible types compared to the type
hints in your code. To test this locally, run the type check test in the top-level directory of the repository.

To find incorrectly used types by type checking in `mypy`, use
```bash
make check-types
```

If you aren't presented with any errors, then that means your code is ready to commit!

It is recommended to install pre-configured [Git hooks](https://github.com/unitaryfoundation/mitiq/blob/main/.git-hooks/) from Mitiq repository by running `make install-hooks` from the root of the repository immediately after cloning.
In particular, the pre-commit hook will run both `make check-format` and `make check-types` before each commit.

## Proposing a new feature to Mitiq
If you are interested in adding a substantial new feature or functionality to Mitiq, please make a copy of our Request For Comments (RFC) [template](https://docs.google.com/document/d/1adomheXpbqp4YIBFQ49IsAJzuJKWyr75GRO1NeWg0Fo/) and fill out the details of your enhancement proposal.
Take a look at [previous RFCs](#list-of-accepted-rfcs) for examples on how to fill out your proposal.
Once you have completed your proposal, [create a feature request issue](https://github.com/unitaryfoundation/mitiq/issues/new?assignees=&labels=feature-request&template=feature_request.md&title=) and add a link to your proposal document (make sure to enable commenting on the RFC!).
For any part of the template that you weren't able to complete please mention that in the issue description.

### List of accepted RFCs
This is a list of accepted request-for-comments (RFC) documents by date of creation (reverse chronological order):

- [Virtual Distillation RFC](https://docs.google.com/document/d/1833IaqzpJ7DbfasgXt59AA93fJk4_EUDfBV4yNHHyic/edit?usp=sharing) by Chris de Hart, Jari Egbers, Kajetan Knopp and Lara Boloni Jan 31, 2025
- [Probabilistic Error Amplification RFC](https://docs.google.com/document/d/1l-74EFdMA0CSFUpHjqCyQYb3ZKCmY77seB1_mOZo5Co/edit?usp=sharing) by Misty Wahl (@Misty-W) Dec 20, 2024
- [Layerwise Richardson Extrapolation RFC](https://docs.google.com/document/d/1oFRl4wMGMtn57V0c_1egaHh0WUUAbtgW-U_QxNL9_kY/edit?usp=sharing) by Purva Thakre (@purva-thakre) and Vincent Russo (@vprusso) Apr 24, 2024
- [Robust Shadow Estimation](https://docs.google.com/document/d/1B5FnqQDvoRYap5fGPqzcbp-RXIrUFjbBcLiWIUrLmuA) by Min Li (@Min-Li) Jun 16, 2023
- [Error Mitigation by Subspace Expansion](https://docs.google.com/document/d/1JyQAwiw8BRT_oucZ6tQv0id6UhSdd3df1mNSPpOvu1I) by Ammar Jahin, Dariel Mok , Preksha Naik, Abdulrahman Sahmoud (@bubakazouba) Apr 28, 2023
- [Implementation RFC for Mitiq calibration](https://docs.google.com/document/d/1EZUJyEEUQUH33UOgSIzCCvXyxP0WLOQn11W0x4Ox4nY/edit) by Andrea Mari (@andreamari) Nov 2, 2022
- [Calibration tools for error mitigation RFC (abstract general solutions)](https://docs.google.com/document/d/1otUHnTlyNS-0rxGAxltHLF1iD5C9qT9oEZ3jn8VHWgw/edit) by Andrea Mari (@andreamari) Oct 6, 2022
- [Identity insertion scaling RFC](https://docs.google.com/document/d/1hbd9frjYiSy0WujA0iCccc-oMO4Q-kZc2G4b3lkJHdk/edit) by Purva Thakre (@purva-thakre) Jun 29, 2022
- [Readout Confusion Inversion RFC](https://docs.google.com/document/d/1buO5PrO5sS02VXjcaYf37RuR0rF6xpyr4J9H1tI4vN4/edit) by Amir Ebrahimi (@amirebrahimi) Jun 16, 2022
- [Documentation reorganization RFC](https://docs.google.com/document/d/13un5TZPknSOhmOBkrL2rsofjGfdp2jDnd-DywLpGFPc/edit) by Ryan LaRose (@rmlarose) Dec 1, 2021
- [Learning-based PEC RFC](https://docs.google.com/document/d/1VItesy6R5SlUa_YXW1km7IjFZ8kzyFeHUepHak1fEh4/edit) by Misty Wahl (@Misty-W) Oct 25, 2021
- [Digital dynamical decoupling RFC](https://docs.google.com/document/d/1cRwFCTn6kUjI1P0kNydtevxIYtE4r8Omd_iWK0Pe8qo/edit) by Aaron Robertson (@Aaron-Robertson) Jan 28, 2021

### Checklist for adding an approved QEM Technique

After your RFC is accepted, the proposed feature (for example, a new QEM Method) will require the following:

- Add the new QEM method to `mitiq/abbreviated_name_of_qem_method` such that the corresponding units tests are in `mitiq/abbreviated_name_of_qem_method/tests`
- The code must follow the formatting and style guidelines discussed [above](#style-guidelines),
- The new module should be added to the [](apidoc.md) using the instructions found in [](contributing_docs.md#automatically-add-information-from-the-api-docs),
- Add documentation for the new QEM method, additional details are available in [](contributing_docs.md#adding-files-to-the-user-guide),
- Update `docs/source/guide/glossary.md` with a one-line summary of what your new feature accomplishes, and
- Update the [](./readme.md#quick-tour) section of the `README.md` with information related to your new technique.


## Code of conduct
Mitiq development abides to the [](./code_of_conduct.md).

## Lifecycle
The basic development workflow for Mitiq is done in units of milestones which are usually one month periods where we focus our efforts on thrusts decided by the development team, alongside community members.
Milestones are tracked using the [GitHub milestone feature](https://github.com/unitaryfoundation/mitiq/milestones) and all issues that are planned to be addressed should be tagged accordingly.

All releases for Mitiq are tagged on the `main` branch with tags for the version number of the release.
Find all the previous releases [here](https://github.com/unitaryfoundation/mitiq/releases).

