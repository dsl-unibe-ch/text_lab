# Visualize Data

!!! note "Draft for the new interface"
    This guide describes the Visualize Data page of the new Text Lab
    interface. It replaces [the current guide](../visualize_data.md) when
    the new interface goes live.

Visualize Data turns a question about your data into charts and
statistical tests. You upload a table, describe in plain language what you
want to see, and a team of AI agents draws the charts, runs the tests and
summarises the findings. Every result comes with the code that reproduces
it, in Python and optionally in R.

Typical uses are getting to know a new dataset, checking a hypothesis
quickly, comparing groups, or producing a first version of a figure.

Everything runs on the University's computing cluster. Your data is never
sent to outside services.

## Quick start

1. **Drop your table** on the form: CSV, TSV, Excel or JSON. It is read at
   once, and its columns appear, with what each one holds.
2. **Untick the columns** you do not need (optional). All are ticked at
   first.
3. **Describe what you want** in the request box, or click one of the
   suggested requests, which name your own columns. Leave the box empty
   for a general exploration of the table.
4. **Choose a model**, and tick **Also write R code** if you work in R
   (both optional).
5. Press **Analyze**.

The analysis continues in the background; you can leave the page and come
back. To ask something else about the same table, change the request and
press **Analyze** again: the table stays, and every analysis keeps its own
record under *Your analyses*, so you can compare them.

## Preparing your data

* **Formats:** CSV (`.csv`), TSV (`.tsv`), Excel (`.xlsx`, `.xls`) and
  JSON (`.json`, as a list of records or as JSON Lines). For Excel files,
  the sheet that was active when the file was last saved is used.
* **Layout:** one row per observation, one column per variable, and column
  names in the first row.
* **Size:** up to **300,000 rows** are analysed; any further rows are
  ignored, and the summary says so.

### Tips for good results

* **Use clear column names.** The AI chooses columns by name, so
  `income_chf` works better than `var17`.
* **Store numbers as numbers.** Values such as `12 kg` or `1'200` are read
  as text. Put units in the column name instead.
* **Keep category spellings consistent.** `Female`, `female` and `F` count
  as three different groups.
* **Empty cells are fine.** Rows with missing values are left out of each
  chart or test that uses those columns.

## The table and its columns

Once the table is read, its name, size and number of columns appear, with
two tabs:

* **Columns**: one row per column, with
    * **Type**, as the agents will treat the column: *number*, *date*,
      *category*, or *text* (free text: a column whose entries average
      more than 50 characters, used for word clouds but not for
      statistics);
    * **Filled**, the share of rows with a value;
    * **Distinct**, the number of different values;
    * **Values**, the range and mean of numbers, the first and last date,
      the most frequent categories, or the average length of texts.

    The columns are described from the first 2,000 rows.
* **First rows**: the first 10 rows, to check that the file was read
  correctly.

**Tick the columns to analyse.** All are ticked at first; **All** and
**None** tick or clear them together, and the count shows how many are
ticked. Fewer columns give more focused results, especially for wide
tables. If no column is ticked, all of them are analysed.

To analyse another table, drop it on the form: it replaces the first one.

## Writing a good request

Describe what you want as specifically as you can: name the columns, the
type of chart, and the test if you know it. Under the request box, Text
Lab suggests a few requests that name your own columns, for example
*Is income different between the groups of sex?*; click one to copy it
into the box, and change it as you like. If you leave the box empty, the
agents perform a general exploratory analysis with a few interactive
charts.

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
    Charts are **interactive** by default: you can point at them to read
    values, and zoom and pan in the browser. Ask for **static** or
    **publication-ready** charts to get high-resolution images (PNG,
    300 dpi) for papers and slides. You can ask for both in one request.

## How it works

A **supervisor** agent reads your request and a summary of your columns
(names, types and example categories), and plans the work for up to three
**specialist** agents:

| Specialist | Produces | Used when |
|---|---|---|
| **Interactive** | Interactive charts (Plotly) | By default for every chart |
| **Static** | High-resolution images (Matplotlib/Seaborn) and word clouds | You ask for static or publication figures, a pair plot or a word cloud |
| **Statistics** | Tables with test results | You ask for tests, correlations, associations or regression |

The specialists work at the same time. When they are done, the supervisor
writes the summary. *What the agents can do*, under the request box, sums
this up.

While the agents work, their record under *Your analyses* shows their
steps as they take them, the newest at the foot, for example:

* *Supervisor planned 3 task(s).*: which specialists work on your request.
* *Worker 'static' is running 4 tools: Static Histogram, Static Box Plot,
  ...*: what each specialist produces. Agents with more tools take longer.
* *... failed: ... Retrying...* and *is retrying* (in amber): a chart or
  test did not work at the first attempt, and the agent corrects it. This
  is normal.
* *Worker 'stats' is interpreting the results.*: the statistics specialist
  explains its numbers.

When no ready-made chart fits, an agent writes its own Python code. That
code runs in a sealed-off area of your session that holds only a copy of
your table: it cannot open your other files, your home folder or research
storage, and has no internet access. Code that tries to use files or other
programs is rejected, and the agent rewrites it.

An analysis usually takes one to a few minutes, and is stopped after 10
minutes. The first analysis with a model takes longer, because the model
is loaded into the GPU's memory. If another tool is using the GPU, such as
a transcription, the analysis waits for it: jobs use the GPU one at a
time. **Cancel** stops an analysis after the agents' current step; what
they made until then is not kept.

## Available charts and tests

### Charts

| Chart | Interactive | Static |
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

A correlation heatmap can be limited to some columns, for example "a
heatmap of all columns ending in _mean". When no standard chart fits your
request, the agent writes its own plotting code (*Custom chart*).

### Statistical tests

Which test fits depends on the kind of columns you want to relate:

| Your question involves... | Test | What you get |
|---|---|---|
| Two **numeric** columns | **Correlation** | Strength and direction of the relationship, p-value and confidence interval. Pearson (default), Spearman or Kendall |
| A **numeric** column and **groups** | **Group comparison** | Whether the values differ between the groups, with an effect size and assumption checks (see below) |
| Two **categorical** columns | **Association test** | The crosstab with row percentages, a chi-square test, and Cramér's V for the strength of the association. For a 2×2 table with small counts, Fisher's exact test is used instead |
| A **numeric outcome** and predictors | **Linear regression** | Ordinary least squares with intercept: coefficients with 95% confidence intervals, p-values, R² and adjusted R² |
| A **yes/no outcome** and predictors | **Logistic regression** | Odds ratios with 95% confidence intervals, p-values and McFadden's pseudo R². For numeric predictors also the odds ratio per standard deviation |
| One target and **all** numeric columns | **Correlation ranking** | Which columns are most strongly correlated with the target. Identifier columns such as *id* are left out |

**Group comparisons** come in two versions:

| | Parametric (default) | Non-parametric |
|---|---|---|
| **Two groups** | t-test: Student's t-test when both groups have the same size, otherwise Welch's t-test. Effect size: Cohen's d | Mann-Whitney U test. Effect size: rank-biserial correlation |
| **Three or more groups** | One-way ANOVA with Tukey post-hoc tests. Effect size: partial eta squared | Kruskal-Wallis test with pairwise Mann-Whitney tests (Holm-corrected). Effect size: eta squared (H) |
| **Use for** | Roughly normally distributed measurements | Ratings, Likert scales and ranks, small groups, or skewed data |

Ask for a non-parametric test explicitly ("use a non-parametric test"), or
let the AI choose: every group comparison reports **assumption checks**, a
Shapiro-Wilk normality test per group and Levene's test for equal
variances, and a note when they look violated. When normality looks
doubtful, a parametric comparison also runs the non-parametric test as a
**robustness check**, so you can see whether the result holds without
asking again. Post-hoc tests are shown for up to 10 groups.

**Yes/no outcomes and categories:**

* An outcome with two values, such as *yes/no*, *M/B* or *1/2*, is coded
  as 1 and 0 automatically. The result says which value counts as 1: a
  typical positive label such as *yes* or *M*, otherwise the larger number
  or the alphabetically later value.
* Text predictors in a logistic regression are compared with their
  alphabetically first category (the *reference*).

Rows with missing values in the columns of a test are left out of that
test. Empty columns without a name, which a comma at the end of every line
of a CSV file creates, are ignored; they do not appear among the columns
either.

## Understanding the results

When the analysis is done, its record shows:

* **What was asked**: the request, the columns and the model.
* **The summary**: the supervisor's written summary of all results. The
  numbers in it come from the actual calculations, but the interpretation
  is written by the AI. If a test you asked for was not run, the summary
  says so.
* **Statistics**: one box per test with the result table and its code.
* **Charts**: each chart with its code under *Python code* (and *R code*).
  Interactive charts show values when you point at them, can be zoomed and
  panned, and can be saved as an image with the camera icon in their
  toolbar.
* **How the agents worked**: their steps, folded away.
* **Downloads** (see below).

The records stay until you remove them or your Text Lab session ends.

!!! warning "Check the results"
    The AI can choose an unsuitable chart or test, or misread a result.
    Read the result tables yourself and use the code to verify important
    findings. Group comparisons report assumption checks, but they do not
    choose the test for you, and the other tests do not check their
    assumptions (for example linearity in a regression).

## Reproducible code in Python and R

Every chart and test comes with code that reproduces it on your own
computer: open *Python code* under it, and **Copy** it. Replace the file
name `your_data.csv` in the code with the path to your file.

### R code

Tick **Also write R code** before you press *Analyze* to get *R code*
next to the Python code of each result.

* Charts use **ggplot2**, statistical tests use base R. The first lines of
  each snippet list the packages it needs, with the command to install
  them. R 4.1 or newer is required.
* **Statistical results are the same** in R and Python. The R code repeats
  the choices the Python code made, for example Student's or Welch's
  t-test, or how a yes/no outcome was coded. The only exception is the
  odds ratio of Fisher's exact test, which R estimates slightly differently
  (the p-value is the same).
* **Charts show the same data but look different**, because default bin
  widths, colours and styles differ between the libraries. Sampled pair
  plots and word clouds can differ in detail.
* For interactive charts, the R code contains a commented line
  (`plotly::ggplotly(p)`) that turns the chart into an interactive one.
* **Custom charts have no R version**, because they are made from code the
  AI wrote freely in Python. They say *R code is not available for plots
  made from custom Python code*.

!!! tip "Double-check your results in R"
    Running the R code is a quick way to verify the numbers independently
    of the Python version.

## Downloads

An analysis with charts or statistics offers two downloads:

* **`<table>-report.html`**: a complete report with the request, the
  summary, the statistical results, all charts and their code (including
  R code if you asked for it). It opens in any web browser, also offline.
* **`<table>-analysis.zip`**: the report, each interactive chart as an
  `.html` file that keeps its interactivity, each static chart as a
  `.png` image, and the Python code of each chart as a `.py` file (and its
  R code as an `.R` file if you asked for it). The code of the statistical
  tests is in the report.

## Data privacy and security

* **Processing on the cluster:** your data is analysed on the UBELIX
  compute nodes. It never leaves the University of Bern's network and is
  never sent to external services such as OpenAI, Google or Anthropic.
* **No AI training:** the models only read your data to answer your
  current request. They do not learn from it, and it is never used to
  train or improve them.
* **Temporary private storage:** your table is saved in your session's
  private workspace on the compute node, while you analyse it. Only your
  user account can open it, and nothing is written to your home directory.
  Each analysis works on its own copy, together with the charts it
  produces; the copy is deleted when the analysis ends. The table itself
  is deleted when you drop another one on the form (one you dropped
  before reloading the page stays until you have dropped three more), and
  in any case when your Text Lab session ends.
* **Results for the session:** the results and their downloads stay until
  you remove the analysis or your session ends. Download them to keep
  them.
* **AI-written code:** custom charts are created by code the AI writes. It
  runs sealed off, under your account, and is stopped if it runs longer
  than a minute.

## Troubleshooting

Click a problem to see what to do.

??? question "'... could not be read as a table'"
    Check that the file is a CSV, TSV, Excel or JSON file with the column
    names in its first row, and that an Excel file is not protected with a
    password. Convert unusual formats to CSV.

??? question "'The analysis could not be started. The model produced a tool call Ollama could not parse'"
    Some models occasionally produce instructions that the system cannot
    read, especially with certain data. Choose another model and run the
    analysis again.

??? question "'The analysis took longer than 10 minutes and was stopped'"
    Ask for fewer charts or tests at once, tick fewer columns, or split
    your request into several analyses.

??? question "'The model did not answer within 300 seconds'"
    The model server was busy or stuck. Run the analysis again, or try
    another model. If nothing responds any more, end your Text Lab session
    and start a new one.

??? question "'The language model server does not answer'"
    The service that runs the models is still starting, as it does at the
    beginning of a session. Wait a minute and press *Analyze* again. If it
    keeps failing, send `ollama.log` from the session's folder to support.

??? question "I got fewer charts than I asked for"
    List each chart you want explicitly, for example "1) a histogram of
    age, 2) a box plot of income by sex". Very long requests are more
    reliable when split into several analyses.

??? question "The AI used the wrong columns"
    Tick only the relevant columns, and use the exact column names in your
    request.

??? question "A test on a text column fails"
    Columns of type *text* (long texts such as comments) cannot be used for
    statistics; use them for word clouds. For two category columns (such
    as *sex* and *smoker*), ask for an association test instead of a
    correlation or t-test.

??? question "The logistic regression reports 'separation' or extremely large odds ratios"
    If the result reports separation, one predictor (almost) perfectly
    predicts the outcome, so its effect cannot be estimated. Remove that
    predictor, or merge rare categories of it.

    Without that note, a huge odds ratio usually comes from a predictor
    measured on a small scale: the odds ratio is per one-unit increase,
    and a predictor that only ranges from 0 to 0.3 never increases by one
    unit. Read the **OR per SD** column instead, the odds ratio per
    standard deviation of the predictor.

??? question "'Some expected counts are below 5'"
    Some combinations of categories are too rare for a reliable chi-square
    test. Merge rare categories (for example combine small faculties into
    *Other*) and run the test again.

??? question "The pair plot says '(sampled 5,000 rows)'"
    Pair plots of very large datasets are drawn from a random sample of
    5,000 rows to keep them readable and fast. The statistics are always
    computed on all rows.

??? question "A chart says 'The chart could not be drawn'"
    The browser could not draw the interactive chart. The report and the
    ZIP among the downloads hold it.

??? question "A chart says 'R code is not available'"
    The chart was made with custom code. Ask for one of the standard chart
    types (see [Available charts and tests](#available-charts-and-tests))
    to get R code.

??? question "The R code fails with 'could not find function'"
    Install the packages listed at the top of the snippet with the
    `install.packages(...)` line shown there, and check that you use R 4.1
    or newer.

??? question "The R results differ from the Python results"
    Check that you use the same file. If your data has more than 300,000
    rows, keep the `head(df, 300000)` line in the R code, because Text Lab
    only analysed those rows. If the numbers of a test still differ,
    contact the Data Science Lab with both results.

??? question "The table looks wrong under *First rows*"
    Check that the first row contains the column names, and that an Excel
    file was saved with the correct sheet active. Convert unusual formats
    to CSV.

For help analysing your data, contact the Data Science Lab (DSL).
