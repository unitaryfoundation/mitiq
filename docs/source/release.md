# Releasing a new version of Mitiq

```{note}
These instructions are for Mitiq maintainers.
Releasing a new version of Mitiq is typically performed by the milestone manager whose responsibilities are detailed [here](./milestone_managing).
```

When the time is ready for a new release, follow the checklist and
instructions of this document to go through all the steps below:

```{contents}
   :local:
   :depth: 3
```

## Prepare the main branch

The start of any release is drafting the changelog and bumping the version number.
Ensure the commit where these changes are made include authorship for all contributors for the given milestone (code, or not).
(Co-authored commits can be created by following the documentation [here](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors).)

### Update the changelog

This task has two parts:

1.  Make sure that `CHANGELOG.md` has an entry for each pull request
    (PR) since the last release. This can be generated from the commit
    history using `git log vX.Y.Z.. --pretty=format:"- %s [%an]"`
    where `vX.Y.Z` is the last version of Mitiq which was released.
    The author names need to then be replaced with the author's GitHub
    handle. An example might look like `- Update python-rapidjson requirement from <=1.6 to <1.8 (#1389) [@dependabot[bot]]`
    once completed.

    ```{tip}
    Alternatively, the list of released changes can be generated via [GitHub CLI](https://cli.github.com/) with the following commands:
        
        LATEST_TAG=$(gh release list --repo unitaryfoundation/mitiq --limit 1 --json tagName --jq '.[0].tagName')
        gh api repos/unitaryfoundation/mitiq/compare/$LATEST_TAG...main --paginate --jq '.commits | reverse | .[] | "- " + (.commit.message | split("\n")[0]) + " [@" + .author.login + "]"'
    This method requires installing (and authenticating on) the Github CLI, but has the advantage that the output list comes already with Github handles, hence removing a tedious step for the release manager.    
    ``` 
2.  The release manager should add a "Summary" section with a couple
    sentences describing the latest release, and then update the title
    of the release section to include the release date and remove the
    "In Development" designation.

### Bump version in VERSION.txt

When releasing a new version, one must update the `VERSION.txt` file
which is the single source of truth for version information. We follow
[SemVer](https://semver.org/), so typically a release will involve changing the version from
`vX.Y.Zdev` (development) to `vX.Y.Z` (released).

### Create a release pull request

A pull request with the changes mentioned above should be created against the _main_ branch as part of the release preparation. 
The pull request must be reviewed by at least one Mitiq maintainer in addition to the milestone manager, 
and its title should be "X.Y.Z Release", where `X.Y.Z` represents the version number to be released.

Note: This pull request triggers a specialized build workflow due to the keyword 'release' in its title. 
This includes additional steps, such as running a link checker to verify all links within the documentation pages.

## Do the release

Once the above changes in the `CHANGELOG.md` and `VERSION.txt` are merged into main, you are ready to do the release.

### Create a new tag

Once the above changes (new changelog and new version) are merged into
the main branch, checkout and pull the latest on the main branch
from your local machine. Then once you are up to date, tag the most
recent commit on main (using `git tag`) with a tag that matches the
number `VERSION.txt` (with a preceding "v", so `0.1.0` is `v0.1.0`)
and push this tag to the Github repository.

```bash
git tag v0.1.0
git push origin v0.1.0
```

### Release the new version on Github

```{note}
You need to have write access to the Mitiq Github repository to make a
new release.
```

There should be a new draft release on GitHub created by the
[gh-release](https://github.com/unitaryfoundation/mitiq/blob/main/.github/workflows/gh-release.yml) action, triggered by the tag you made in the
previous step [here](https://github.com/unitaryfoundation/mitiq/releases). You
will need to review it and publish the release.

- GitHub will create compressed files with the repository.
- GitHub adds the full changelog in the draft release. Please keep the content related to the new release and remove the content related to previous releases.

```{note}
If all the above steps have been successfully completed, ReadTheDocs
(RTD) will automatically build new `latest` and `stable` versions of the
documentation. So, no additional steps are needed for updating RTD. You
can verify changes have been updating by viewing
<https://mitiq.readthedocs.io/>. Note that this may require a
significant amount of time. You can check the build status
[here](https://readthedocs.org/projects/mitiq/builds/)
```

### Release the new version on PyPI

Once the GitHub release is published, the release is also published on
PyPI by the [publish-pypi](https://github.com/unitaryfoundation/mitiq/blob/main/.github/workflows/publish-pypi.yml) action. This may require a few
minutes. If it seems like it didn't push a new version to PyPI, you can
trigger it manually. Go to
<https://github.com/unitaryfoundation/mitiq/actions/workflows/publish-pypi.yml>
and use the "Run Workflow" button to publish the new version on PyPI.

In case the action for releasing on PyPI fails, the Python commands to
release Mitiq are:

```bash
uv build
uv publish
```

```{note}
You need to be a registered maintainer of Mitiq project on PyPI to
upload a new release on PyPI from your local machine as a token is 
required for authentication.
```

## Update the new development version

Add a new section to the `CHANGELOG.md` to track changes in the following release, meaning that if `vX.Y.Z` was just released, then there should be a section for `vX.(Y+1).0` that is marked "In Development".
Finally, change the version in `VERSION.txt` from `vX.Y.Z` to `vX.(Y+1).0dev`.
