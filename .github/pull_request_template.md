Description
-----------

Describe your pull request (PR) here. Thank you for contributing to Mitiq! 🙂

Checklist
-----------

Check off the following once complete (or if not applicable). The PR will be reviewed once this
checklist is complete and all tests are passing.

- [ ] I added unit tests for new code.
- [ ] I used [type hints](https://www.python.org/dev/peps/pep-0484/) in function signatures.
- [ ] I used [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) docstrings for functions.
- [ ] I [updated the documentation](../blob/master/docs/CONTRIBUTING_DOCS.md) where relevant.
- [ ] I updated the [changelog](https://github.com/unitaryfund/mitiq/blob/master/CHANGELOG.md) including author and PR number (@username, gh-xxx)

If some items remain, you can mark this a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/).

Tips
----

- If the validation check fails:
  1. Run `make check-style` (from the root directory
  of the repository) and fix any [flake8](http://flake8.pycqa.org) errors.
  2. Run `make format` to format your code with the [black](https://black.readthedocs.io/en/stable/index.html)
  autoformatter.
- Write "Fixes #XYZ" in the description if this PR fixes Issue #XYZ.