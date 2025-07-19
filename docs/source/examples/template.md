---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# An example Jupyter Notebook

This notebook is a demonstration of directly-parsing Jupyter Notebooks into
Sphinx using the MyST parser.

## Markdown


### Configuration

https://myst-parser.readthedocs.io/en/latest/using/intro.html#getting-started

To build documentation from this notebook, the following options are set:

```python
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")
```

### Syntax

As you can see, markdown is parsed as expected. Embedding images should work as expected.
For example, here's the MyST-NB logo:

```md
![myst-nb logo](../img/unitary_foundation_logo.png)
```

![myst-nb logo](../img/unitary_foundation_logo.png)

By adding `"html_image"` to the `myst_enable_extensions` list in the sphinx configuration ([see here](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#html-images)), you can even add HTML `img` tags with attributes:

```html
<img src="../img/unitary_foundation_logo.png" alt="logo" width="200px" class="shadow mb-2">
```

<img src="../img/unitary_foundation_logo.png" alt="logo" width="200px"  class="shadow mb-2">

Because MyST-NB is using the MyST-markdown parser, you can include rich markdown with Sphinx in your notebook.
For example, here's a note admonition block:

:::::{note}
**Wow**, a note!
It was generated with this code ([as explained here](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#html-admonitions)):

````md
:::{note}
**Wow**, a note!
:::
````

:::::

If you wish to use "bare" LaTeX equations, then you should add `"amsmath"` to the `myst_enable_extensions` list in the sphinx configuration.
This is [explained here](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#direct-latex-math), and works as such:

```latex
\begin{equation}
\frac {\partial u}{\partial x} + \frac{\partial v}{\partial y} = - \, \frac{\partial w}{\partial z}
\end{equation}

\begin{align*}
2x - 5y &=  8 \\
3x + 9y &=  -12
\end{align*}
```

\begin{equation}
\frac {\partial u}{\partial x} + \frac{\partial v}{\partial y} = - \, \frac{\partial w}{\partial z}
\end{equation}

\begin{align*}
2x - 5y &=  8 \\
3x + 9y &=  -12
\end{align*}

Also you can use features like **equation numbering** and referencing in the notebooks:

```md
$$e^{i\pi} + 1 = 0$$ (euler)
```

$$e^{i\pi} + 1 = 0$$ (euler)

Euler's identity, equation {math:numref}`euler`, was elected one of the
most beautiful mathematical formulas.

You can see the syntax used for this example [here in the MyST documentation](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html).

## Code cells and outputs

You can run cells, and the cell outputs will be captured and inserted into
the resulting Sphinx site.

### `__repr__` and HTML outputs

For example, here's some simple Python:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
data = np.random.rand(3, 100) * 100
data[:, :10]
```

This will also work with HTML outputs

```{code-cell} ipython3
import pandas as pd
df = pd.DataFrame(data.T, columns=['a', 'b', 'c'])
df.head()
```

as well as math outputs

```{code-cell} ipython3
from IPython.display import Math
Math(r"\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}")
```

This works for error messages as well:

```{code-cell} ipython3
:tags: [raises-exception]

print("This will be properly printed...")
print(thiswont)
```

### Images

Images that are generated from your code (e.g., with Matplotlib) will also
be embedded.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(*data, c=data[2])
```

## Thumbnail for the Notebook

To add a thumbnail for an example notebook, first add the thumbnail image file to `docs/source/_thumbnails`. Next, modify the `docs/source/conf.py` to include the example and thumbnail in the nbsphinx_thumbnails dictionary at the end of the file. The sample below contains both a generic template and an actual example.
```{code-cell} ipython3
nbsphinx_thumbnails = {
    "examples/{EXAMPLE_FILENAME_WITHOUT_.md}": "_static/{THUMBNAIL_FILENAME_WITH_EXTENSION}",
    "examples/hamiltonians": "_static/vqe-cirq-pauli-sum-mitigation-plot.png"
    }
```
