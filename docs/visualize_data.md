# Visualize Data

Visualize Data turns a question about your data into plots and statistical tests. You upload a table, describe in plain language what you want to see, and a team of AI agents creates the charts, runs the tests and summarises the findings. Every result comes with the code that reproduces it, in Python and optionally in R.

Typical uses are getting to know a new dataset, checking a hypothesis quickly, comparing groups, or producing a first version of a figure.

All processing runs on the university cluster. Your data is never sent to an external service.

## Quick start

1. **Choose a model** in the sidebar under *Select Analysis Model*.
2. **Upload your data** as a CSV, TSV, Excel or JSON file.
3. **Select the columns** to analyse. All columns are selected by default.
4. **Describe what you want** in the text box, or leave it empty for a general exploratory analysis.
5. Tick **Also generate equivalent R code** if you work in R (optional).
6. Click **Generate Visualisations** and follow the progress in the *Agent Activity Log*.
7. **Explore the results** and download them with the **Download ... (.zip)** button.

Running again replaces the previous results on the page, and uploading a different file clears them, so download anything you want to keep first.

## Preparing your data

* **Formats:** CSV (`.csv`), TSV (`.tsv`), Excel (`.xlsx`, `.xls`) and JSON (`.json`, as a list of records or as JSON Lines). For Excel files, the sheet that was active when the file was last saved is used.
* **Layout:** one row per observation, one column per variable, and column names in the first row.
* **Size:** up to **300,000 rows** are analysed; any further rows are ignored, and the summary says so.
* **Column types** are detected automatically: numbers, categories, dates, and free text. A column whose entries average more than 50 characters is treated as free text, which is used for word clouds but not for statistics.

### Tips for good results

* **Use clear column names.** The AI chooses columns by name, so `income_chf` works better than `var17`.
* **Store numbers as numbers.** Values such as `12 kg` or `1'200` are read as text. Put units in the column name instead.
* **Keep category spellings consistent.** `Female`, `female` and `F` count as three different groups.
* **Empty cells are fine.** Rows with missing values are left out of each plot or test that uses those columns.

## Preview and column selection

After uploading, open **Preview Data** to check that the file was read correctly:

* **Raw Data**: the first 10 rows.
* **Column Profile**: for each column, its type, the share of filled cells, the number of different values, the range (numbers) or most frequent value (text). Based on the first 2,000 rows.

Under **Select columns to include in the analysis**, keep only the columns you are interested in. Fewer columns give more focused results, especially for wide datasets. *Select All* and *Clear All* help; if no column is selected, all columns are used.

## Writing a good request

Describe what you want as specifically as you can: name the columns, the type of plot, and the test if you know it. If you leave the box empty, Text Lab performs a general exploratory analysis with a few interactive plots.

| You want to... | Example request |
|---|---|
| Get an overview | *(leave the box empty)* |
| See a distribution | "Histogram of age for each sex" |
| Explore a relationship | "Scatter plot of income against age, coloured by region, and test the correlation" |
| Compare groups | "Is radius_mean different between diagnosis M and B? Show a box plot and run a t-test." |
| Compare ratings or Likert answers | "Do the faculties differ in satisfaction? Use a non-parametric test." |
| Relate two categories | "Is smoking associated with diagnosis?" |
| Find important variables | "Which columns are most strongly correlated with diagnosis?" |
| Model an outcome | "Linear regression of price on size and number of rooms" |
| Model a yes/no outcome | "Which factors predict dropout? Use logistic regression with age, faculty and grade." |
| Analyse text | "Word cloud of the comments column" |
| Get publication figures | "Static, publication-ready box plot of score by group" |

!!! tip "Interactive or static?"
    Plots are **interactive** by default: you can hover, zoom and pan in the browser. Ask for **static** or **publication-ready** plots to get high-resolution images (PNG, 300 dpi) for papers and slides. You can ask for both in one request.

## How it works

A **supervisor** agent reads your request and a summary of your columns (names, types and example categories), and plans the work for up to three **specialist** agents:

| Specialist | Produces | Used when |
|---|---|---|
| **Interactive** | Interactive charts (Plotly) | By default for every plot |
| **Static** | High-resolution images (Matplotlib/Seaborn) and word clouds | You ask for static or publication figures, a pair plot or a word cloud |
| **Statistics** | Tables with test results | You ask for tests, correlations, associations or regression |

The specialists work at the same time. When they are done, the supervisor writes the summary.

The **Agent Activity Log** shows what is happening, for example:

* *Supervisor planned 3 task(s).*: which specialists are working on your request.
* *Worker 'static' is running 4 tools: Static Histogram, Static Box Plot, ...*: what each specialist is producing. Agents with more tools take longer.
* *... failed: ... Retrying...* and *is retrying*: a plot or test did not work on the first attempt, and the agent is correcting it. This is normal.
* *Worker 'stats' is interpreting the results.*: the statistics specialist is explaining its numbers.

An analysis usually takes one to a few minutes, and is stopped after 10 minutes. The first analysis after choosing a model takes longer, because the model is loaded into GPU memory. Click **Cancel Analysis** to stop a running analysis.

## Available plots and tests

### Plots

| Plot | Interactive | Static |
|---|---|---|
| **Histogram** | Yes | Yes, with a density curve |
| **Scatter plot** | Yes, optionally coloured by a column | Yes, optionally coloured by a column |
| **Box plot** | Yes | Yes |
| **Line plot** | Yes, connects the rows in data order | Yes, mean per x value with a 95% confidence band |
| **Bar chart** | Yes: mean, sum, count or median per group | Yes, with ±1 standard deviation error bars for mean and median |
| **Scatter matrix / pair plot** | Yes | Yes; very large datasets are sampled to 5,000 rows |
| **Correlation heatmap** | Yes (Pearson or Spearman) | Yes (Pearson or Spearman) |
| **Word cloud** | No | Yes |
| **Custom chart** | Yes | Yes |

A correlation heatmap can be limited to some columns, for example "a heatmap of all columns ending in _mean". When no standard plot fits your request, the agent writes its own plotting code (*Custom chart*).

### Statistical tests

Which test fits depends on the kind of columns you want to relate:

| Your question involves... | Test | What you get |
|---|---|---|
| Two **numeric** columns | **Correlation** | Strength and direction of the relationship, p-value and confidence interval. Pearson (default), Spearman or Kendall |
| A **numeric** column and **groups** | **Group comparison** | Whether the values differ between the groups, with an effect size and assumption checks (see below) |
| Two **categorical** columns | **Association test** | The crosstab with row percentages, a chi-square test, and Cramér's V for the strength of the association. For a 2×2 table with small counts, Fisher's exact test is used instead |
| A **numeric outcome** and predictors | **Linear regression** | Ordinary least squares with intercept: coefficients, p-values and R² |
| A **yes/no outcome** and predictors | **Logistic regression** | Odds ratios with 95% confidence intervals, p-values and McFadden's pseudo R² |
| One target and **all** numeric columns | **Correlation ranking** | Which columns are most strongly correlated with the target |

**Group comparisons** come in two versions:

| | Parametric (default) | Non-parametric |
|---|---|---|
| **Two groups** | t-test: Student's t-test when both groups have the same size, otherwise Welch's t-test. Effect size: Cohen's d | Mann-Whitney U test. Effect size: rank-biserial correlation |
| **Three or more groups** | One-way ANOVA with Tukey post-hoc tests. Effect size: partial eta squared | Kruskal-Wallis test with pairwise Mann-Whitney tests (Holm-corrected). Effect size: eta squared (H) |
| **Use for** | Roughly normally distributed measurements | Ratings, Likert scales and ranks, small groups, or skewed data |

Ask for a non-parametric test explicitly ("use a non-parametric test"), or let the AI choose: every group comparison reports **assumption checks**, a Shapiro-Wilk normality test per group and Levene's test for equal variances, and a note when they look violated. Post-hoc tests are shown for up to 10 groups.

**Yes/no outcomes and categories:**

* An outcome with two values, such as *yes/no*, *M/B* or *1/2*, is coded as 1 and 0 automatically. The result says which value counts as 1: a typical positive label such as *yes* or *M*, otherwise the larger number or the alphabetically later value.
* Text predictors in a logistic regression are compared with their alphabetically first category (the *reference*).

Rows with missing values in the columns of a test are left out of that test.

## Understanding the results

* **Analysis Summary**: the supervisor's written summary of all results. The numbers in it come from the actual calculations, but the interpretation is written by the AI.
* **Statistical Analysis Results**: one card per test with the result table and its code.
* **Generated Visualisations**: each chart with its code under *View Source Code*. Interactive charts can be zoomed and panned, show values when you hover over them, and can be saved as an image with the camera icon in their toolbar.

!!! warning "Check the results"
    The AI can choose an unsuitable plot or test, or misread a result. Read the result tables yourself and use the code to verify important findings. Group comparisons report assumption checks, but they do not choose the test for you, and the other tests do not check their assumptions (for example linearity in a regression).

## Reproducible code in Python and R

Every plot and test comes with code that reproduces it on your own computer. Replace the file name `your_data.csv` in the code with the path to your file.

### R code

Tick **Also generate equivalent R code** before you click *Generate Visualisations* to get R code next to the Python code. The code areas then show a **Python** and an **R** tab.

* Plots use **ggplot2**, statistical tests use base R. The first lines of each snippet list the packages it needs, with the command to install them. R 4.1 or newer is required.
* **Statistical results are the same** in R and Python. The R code repeats the choices the Python code made, for example Student's or Welch's t-test, or how a yes/no outcome was coded. The only exception is the odds ratio of Fisher's exact test, which R estimates slightly differently (the p-value is the same).
* **Plots show the same data but look different**, because default bin widths, colours and styles differ between the libraries. Sampled pair plots and word clouds can differ in detail.
* For interactive plots, the R code contains a commented line (`plotly::ggplotly(p)`) that turns the plot into an interactive one.
* **Custom charts have no R version**, because they are made from code the AI wrote freely in Python. They show *R code is not available for plots made from custom Python code*.

!!! tip "Double-check your results in R"
    Running the R code is a quick way to verify the numbers independently of the Python version.

## Downloading the results

The **Download ... (.zip)** button below the results gives you a ZIP archive with:

* `report.html`: a complete report with the summary, the statistical results, all charts and their code (including R code if you asked for it). It opens in any web browser, also offline.
* Each interactive chart as an `.html` file that keeps its interactivity, and each static chart as a `.png` image.
* The Python code of each chart as a `.py` file, and the R code as an `.R` file if you asked for it.

The code of the statistical tests is included in `report.html`.

## Data privacy and security

* **Processing on the cluster:** your data is analysed on the UBELIX compute nodes. It never leaves the University of Bern's network and is never sent to external services such as OpenAI, Google or Anthropic.
* **No AI training:** the models only read your data to answer your current request. They do not learn from it, and it is never used to train or improve them.
* **Temporary private storage:** for the analysis, your file is saved in a temporary folder in your session's private workspace on the compute node, together with the charts it produces. Only your user account can open it, and nothing is written to your home directory. The folder is **deleted automatically when the analysis finishes**; if an analysis is interrupted, it is deleted when your Text Lab session ends.
* **Results in your browser session:** the results stay on the page until you run a new analysis, upload a different file, reload the page or close the tab. Download them to keep them.
* **AI-written code:** custom charts are created by code the AI writes. It runs on the cluster under your account and is stopped if it runs longer than a minute.

## Troubleshooting

Click a problem to see what to do.

??? question "'The analysis could not be started. The model produced a tool call Ollama could not parse'"
    Some models occasionally produce instructions that the system cannot read, especially with certain data. Choose another model in the sidebar and run the analysis again.

??? question "'Analysis exceeded the 10-minute limit'"
    Ask for fewer plots or tests at once, select fewer columns, or split your request into several runs.

??? question "'The model did not answer within 300 seconds'"
    The model server was busy or stuck. Run the analysis again, or try another model. If nothing responds any more, end your Text Lab session and start a new one.

??? question "I got fewer plots than I asked for"
    List each plot you want explicitly, for example "1) a histogram of age, 2) a box plot of income by sex". Very long requests are more reliable when split into several runs.

??? question "The AI used the wrong columns"
    Select only the relevant columns under *Select columns to include in the analysis*, and use the exact column names in your request.

??? question "A test on a text column fails"
    Free-text columns (long texts such as comments) cannot be used for statistics; use them for word clouds. For two category columns (such as *sex* and *smoker*), ask for an association test instead of a correlation or t-test.

??? question "The logistic regression reports 'separation' or extremely large odds ratios"
    One predictor (almost) perfectly predicts the outcome, so its effect cannot be estimated. Remove that predictor, or merge rare categories of it.

??? question "'Some expected counts are below 5'"
    Some combinations of categories are too rare for a reliable chi-square test. Merge rare categories (for example combine small faculties into *Other*) and run the test again.

??? question "The pair plot says '(sampled 5,000 rows)'"
    Pair plots of very large datasets are drawn from a random sample of 5,000 rows to keep them readable and fast. The statistics are always computed on all rows.

??? question "The R tab says 'R code is not available'"
    The chart was made with custom code. Ask for one of the standard plot types (see [Available plots and tests](#available-plots-and-tests)) to get R code.

??? question "The R code fails with 'could not find function'"
    Install the packages listed at the top of the snippet with the `install.packages(...)` line shown there, and check that you use R 4.1 or newer.

??? question "The R results differ from the Python results"
    Check that you use the same file. If your data has more than 300,000 rows, keep the `head(df, 300000)` line in the R code, because Text Lab only analysed those rows. If the numbers of a test still differ, contact the Data Science Lab with both results.

??? question "The table looks wrong in *Preview Data*"
    Check that the first row contains the column names, and that an Excel file was saved with the correct sheet active. Convert unusual formats to CSV.

For help analysing your data, contact the Data Science Lab (DSL).
