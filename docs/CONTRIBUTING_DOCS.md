# Contributing to the Documentation
This is the contributors guide for the documentation of Mitiq,
the Python toolkit for implementing error mitigation on quantum computers.

## Requirements
Our documentation is generated with
[Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html).
The necessary packages can be installed, from the root Mitiq directory with the command
```bash
make install
```

### Sphinx extensions used to build the docs
- [`myst-nb`](https://myst-nb.readthedocs.io/en/latest/) and [`myst-parser`](https://myst-parser.readthedocs.io/en/latest/) allow both markdown and jupyter notebooks to be included and run by the Sphinx build. Also adds support for [MyST markdown](https://myst-parser.readthedocs.io/en/latest/using/syntax.html) spec.
- [`sphinxcontrib-bibtex`](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/) allows to include citations in a `.bib` file.
- [`pybtex`](https://pybtex.org/) allows to customize how citations are rendered, e.g., APS-style.
- [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io/en/latest/) allows to easily copy-paste code snippets from examples.
- [`sphinx-autodoc-typehints`](https://pypi.org/project/sphinx-autodoc-typehints/) allows to control how annotations are displayed in the API-doc part of the documentation, integrating with  [`sphinx-autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and [`sphinx-napoleon`](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/) which handle specific styling requirements for the API documentation comments.

You can check that Sphinx is installed with `uv run sphinx-build --version`.

### The configuration file
Since the documentation is already created, you need not to generate a
configuration file from scratch (this is done with `uv run sphinx-quickstart`).
Meta-data, extensions and other custom specifications are accounted for
in the `conf.py` file.

### Add/change Sphinx features in the conf.py file

To add specific feature to the documentation, Sphinx extensions can be added to the build.
As an example, to add classes and functions to the API doc, make sure that autodoc
extension is enabled in the `conf.py` file.

```python
extensions = ['sphinx.ext.autodoc']
```

## Updating the Documentation

Documentation is found in `docs/source`, and is divided into the following sections:

-  a **guide**, whose content needs to be written from scratch for new features. For more details, go to [](contributing.md#proposing-a-new-feature-to-mitiq) and [](contributing_docs.md#adding-files-to-the-user-guide)
- **examples** which can be either a jupyter notebook or a MyST formatted notebook. A checklist is available in [](contributing_docs.md#adding-example-files)
- an **API-doc** part, which is (mostly) automatically generated. See [](contributing_docs.md#automatically-add-information-from-the-api-docs) for additional information.

Information in the docs should be added as markdown files using the MyST markdown syntax.
If you are adding a new file (as opposed to editing an existing one), ensure to add it to an associated TOC so that it is discoverable.

The main table of contents (TOC) file for the docs is `index.md`. It includes `guide/guide.md` and `apidoc.md`, among other files. To add a new file to the base TOC, make sure it gets listed in the `toctree` directive like this:
````
```{toctree}
---
maxdepth: 2 
caption: Contents
---
file.md
```
````

```{tip}
If you use VS Code as your text editor there is a nice extension that does syntax highlighting for MyST: [https://marketplace.visualstudio.com/items?itemName=ExecutableBookProject.myst-highlight](https://marketplace.visualstudio.com/items?itemName=ExecutableBookProject.myst-highlight)
```

### Including other files in the docs 

To include `.md` files outside of the documentation `source` directory, you can add a stub `*.md` file to the toctree inside the `docs/source` directory that contains:

````
```{include} path/to/file.md
:relative-docs: docs/
:relative-images:
```
````

where `file.md` is the one to be added. For more information on including files external to the docs, see the [MyST docs](https://myst-parser.readthedocs.io/en/latest/).



### Adding files to the user guide

To add information in the guide, please add markdown (`.md`) files to the `docs/guide` directory.
Remember to add new files to the guide's TOC file `docs/source/guide/guide.md`.

#### User Guide Outline

The different sections of a User's guide comprises of:

- How do I use `new_QEM_method` labeled as `new_QEM_method-1-intro.md`?
- When should I use `new_QEM_method` labeled as `new_QEM_method-2-use-case.md`?
- What additional options are available in `new_QEM_method` labeled as `new_QEM_method-3-options.md`?
- What happens when I use `new_QEM_method` labeled as `new_QEM_method-4-low-level.md`?
- What is the theory behind `new_QEM_method` labeled as `new_QEM_method-5-theory.md`?

The main landing page will link all the sections of the User's Guide such that there is a [workflow diagram](https://mitiq.readthedocs.io/en/latest/contributing_docs.html#adding-workflow-images-to-the-user-guide) for the `new_QEM_method` on this page alongside
links to any tutorials utilizing `new_QEM_method` in `docs/source/examples`. 

```{code-block}
   # Full Name of New QEM Method

   One line summary of what the technique does

   <start_backticks_code_block>{figure} ../img/new_qem_method_workflow.png
   ---
   width: 700px
   name: new_qem_method_workflow_overview
   ---
   The new_QEM workflow in Mitiq is fully explained in the {doc}`new_QEM_method-4-low-level.md` section.
   <end_backticks_code_block>

   Below you can find sections of the documentation that address the following questions:

   <start_backticks_code_block>{toctree}
   ---
   maxdepth: 1
   ---
   new_QEM_method-1-intro.md
   new_QEM_method-2-use-case.md
   new_QEM_method-3-options.md
   new_QEM_method-4-low-level.md
   new_QEM_method-theory.md
   <end_backticks_code_block>

   A simple tutorial using  `new_QEM_method` can be found <insert_link>
```

#### Adding workflow images to the user guide

To insert a workflow to the user's guide of some new technique, a template is available (shown below as a `png`). This template is also available in `svg` format at [mitiq_template.svg](../source/img/general_template.svg).

The `svg` will require edits to the items listed below:

- One word description of the first step of the new technique in `(Edit Step 1)`. For example, if the new technique
requires sampling of the input quantum program, you will edit the text to `1. Sampling`.
- Make changes to `Description of circuit generated by mitiq` to instead briefly provide information about how mitiq is manipulating the input circuit. 
- Feel free to change the color scheme in `mitiq.new_technique` of the `svg` file.

```{figure} ../source/img/mitiq_workflow_template.png
---
name: fig-overview-template
---
The diagram shows the workflow template of a new technique in Mitiq.
```


### Adding code examples

All code examples, besides explanations on the use of core software package features, live in the `examples` directory under `docs/source`. You can add
Jupyter notebooks (`.ipynb`) or MyST markdown notebooks, but MyST formatting will be preferred as it is much easier to diff in version control.

If you have a notebook you want to add, and want to automatically convert it from the `.ipynb` to `.md`, you can use a great Python command line tool called [jupytext](https://jupytext.readthedocs.io/en/latest/index.html).
To convert from an IPython notebook to markdown file, run `jupytext your_filename.ipynb --to myst` and find the converted file at `your_filename.md`.

Further, not only can `jupytext` convert between the formats on demand, but once you install it, you can configure it to manage _both_ a Jupyter and Markdown version of your file, so you don't have to remember to do conversions (for more details, see the `jupytext` docs on [paired notebooks](https://jupytext.readthedocs.io/en/latest/index.html#paired-notebooks)).
Using the paired notebooks you can continue your development in the notebooks as normal, and just commit to git the markdown serialized version when you want to add to the docs.
You can even add this tool as a [git pre-commit hook](https://jupytext.readthedocs.io/en/latest/using-pre-commit.html) if you want!

```{note}
Code cells in notebooks are executed and tested when building the documentation. If any additional Python libraries are needed to execute the notebook, they must be added to [pyproject.toml](https://github.com/unitaryfoundation/mitiq/blob/main/pyproject.toml) in the `docs` section of `[dependency-groups]`.
```

```{tip}
There is a [sample markdown formatted notebook in the `examples` directory](../source/examples/template.md) for you to take a look at as you write your own!
```

### Adding example files

If you are adding a new tutorial to the [documentation examples](../source/examples/examples.md),

- add the file as a `.md` file in `docs/source/examples`. A template is available at [here](../source/examples/template.md).
- add your tutorial in `docs/source/examples/examples.md` such that it shows up in a docs build 
- add a thumbnail image as `docs/source/_thumbnails/your_image.png` and list it in the `nbsphinx_thumbnails` block of `docs/source/conf.py`

### Automatically add information from the API docs

New modules, classes and functions can be added by listing them in the appropriate file (such as `apidoc.md` or a child), e.g.,

````
## New Module
```{eval-rst}
.. automodule:: mitiq.new_module
   :members:
```
````
will add all elements of the `mitiq.new_module` module with a subtitle "New Module." 
You can hand-pick classes and functions to add, to comment them, as well as exclude them.

```{warning}
The `eval-rst` directive must be used for now as [myst-parser](https://github.com/executablebooks/MyST-Parser) only supports parsing docstrings as RST.
Once <https://github.com/executablebooks/MyST-Parser/issues/228> is closed, we can migrate to markdown docstrings.
```

```{tip}
If you are adding new features to Mitiq, make sure to add API docs in the
source code, and to the API page `apidoc.md`.
```

### Adding references 

To add references to the [Mitiq bibliography](bibliography.md), the first step is to add the reference to `docs/source/refs.bib` which is organized alphabetically. For formatting, please see BibTeX documentation for [articles](https://www.bibtex.com/e/article-entry/), [books](https://www.bibtex.com/e/book-entry/), and [others](https://www.bibtex.com/e/entry-types/).

Once the reference has been added to the `docs/source/refs.bib` file, cite the reference in the file by using:
```md
{cite}`title of entry`
```

## Build the documentation locally
The easiest way to build the docs is to run `make docs` from the project 
root directory, which builds the html docs output.

```{tip}
If you want a fresh build with no caching, run `make docs-clean`!
```

To call sphinx directly, `cd` into the `docs` directory and run
```bash
sphinx-build -b html source build
```

These commands generate the `docs/build` folder, which is ignored by the `.gitignore` file.
Once the documentation is generated you can view it by opening it in your browser.

## Testing the Documentation 

When writing a new code example in the docs, you can use different directives 
to include code blocks. 

### Just the code, don't evaluate
If you want to include a code snippet that doesn't get run (but has syntax
highlighting), use the `code-block` directive:

````
```{code-block} python
   1+1        # simple example
```
````

## View the documentation from a PR build

To preview the documentation (mitiq.readthedocs.io) from a specific build in a PR, click `Details` on the `docs/readthedocs.org:mitiq` line of the pull request's merge box in the PR's Conversation timeline. It may be necessary to scroll down to find the `docs/readthedocs.org:mitiq` line.

![RTD of PR](./source/img/read_the_docs_pr.png)

If the Read the Docs build failed, it is still possible to see the documentation from the build by clicking `Details` and then clicking on the `View docs` link on the right side of screen under the time it took for the build to complete.

![RTD failed](./source/img/read_the_docs_failed.png)

## Additional information
[Here](https://github.com/nathanshammah/scikit-project/blob/master/5-docs.md)
are some notes on how to build docs.

[The MyST syntax guide](https://myst-parser.readthedocs.io/en/latest/using/syntax.html) is a
cheat sheet for the extended Markdown formatting that applies to both Markdown files as well as Markdown in Jupyter notebooks.

[The MyST-NB Notebook guide](https://myst-nb.readthedocs.io/en/latest/authoring/basics.html?highlight=markdown#myst-markdown) can help you get you write or convert your notebook content for the docs.
