---
title: COVID-19 Interactive Visualization
layout: single
header:
  image: /assets/images/covid-19-post-assets/covid.jpg
  caption: "Photo credit: [**IRF Coronavirus Rehabilitation Tools**](http://www.rehabforum.org/tools.html)"
classes: wide
toc: true
custom_css: plotly
---

**Last Updated: 31/05/2020**

# What is this post about

In this post I'll present different visuals of the novel *Corona virus (COVID-19)*, to show how it impacted the whole world disproportionately, and how contagious it is.

I'll use dataset from the [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) which contains daily aggregated data, and time series data.

The data provides us with the following four variables:

- **Active**: the number of active cases, cases which have test positive for the virus.

- **Recovered**: number of recovered cases from the virus.

- **Deaths**: number of deaths.

- **Confirmed**: the sum of the previous the three variables.

These variables are provided for each country on a daily basis.

I'll also use two data sources from the [World Bank Group](https://www.worldbank.org/):

- [Total population](https://data.worldbank.org/indicator/SP.POP.TOTL).

- [Population ages 65 and above as a percentage of the total population](https://data.worldbank.org/indicator/SP.POP.65UP.TO.ZS).

The population data might not be very accurate, as it was last updated in 2018, but that won't be a problem.

# Overall statistics pie chart

The following pie chart shows the percentages of the three measures: active cases, recovered cases, and total deaths.

At first sight, this chart comforts us! the virus doesn't seem to be very *fatal*, only about 6% of the total cases are **deaths**, while around 42% had **recovered** from the virus.

{% include covid-charts/overall_stats_pie_chart.html %}

Pie Chart is misleading in interpreting the severity level of the virus, it doesn't show us how fast the virus is spreading, and how the number of deaths is growing *exponentially*.

Later in this post, I'll use different type of charts to answer the previous questions.

# Treemap chart

The following treemap describes the different proportions countries around the world have been affected by this pandemic, grouped together by continent.

{% include covid-charts/most_affected_countries_treemap_chart.html %}

This treemap serves as an overall-look at the total deaths in the most affected countries.

We can see that **Europe** and **North America** constitute around half of the global deaths.

# Most hit countries

After taking a general overview on the virus statistics, and how it has taken down many lives across the world, here in this section I'll address the most hit countries in terms of their deaths tolls.

## Most affected countries pie chart

Looking at the following chart, we can notice that the **U.S.** and the **European** countries are among the most hit countries, and together they constitute *almost* the whole chart.

{% include covid-charts/most_affected_countries_pie_chart.html %}

We can also observe that the chart is divided into three (nearly) equal parts:

- The U.S.

- The U.K., Brazil, and Italy.

- And the remaining eleven countries.

Based on the previous observation, we might ask why the U.S. deaths tolls are so high (as much as eleven countries combined)?

The U.S. is the third in population size (327 million people as of 2018), so we can't compare it with other low-populated countries.

Later in this post, I'll introduce another measures based on the countries populations, to look at the most hit countries from a different perspective.

## Most affected countries bar chart

The previous chart showed us only the deaths tolls, and discarded the other two variables: active cases, and recovery cases.

The following chart explores the three measures for each country.

{% include covid-charts/most_affected_countries_stacked_bar_chart.html %}

This chart helps us understand the relative proportion of each variable compared to the total confirmed cases in a particular country.

One country that stands out among others is **Germany**, while it has as much confirmed cases as other european countries (France, Italy, and Spain), it managed to control the number of active cases, and ended up with the highest recovery rate, mainly because of its excessive testing.

More on Germany's response to the pandemic in this article: [A German Exception? Why the Country’s Coronavirus Death Rate Is Low](https://www.nytimes.com/2020/04/04/world/europe/germany-coronavirus-death-rate.html)

*Note*: Both U.K. and Netherlands have recovery cases, but it's very low, that it's not visible in the chart, quoting the dataset documentation of the recovery cases are computed:

> Recovered cases outside China are estimates based on local media reports, and state and local reporting when available, and therefore may be substantially lower than the true number.

# Growth (spread) of infections over time

## Cases over time

{% include covid-charts/infections_trajectory_line_chart.html %}

## Deaths over time

{% include covid-charts/deaths_trajectory_line_chart.html %}

## Recovery over time

{% include covid-charts/recovery_trajectory_line_chart.html %}

# Choropleth maps

## Global deaths Choropleth map

{% include covid-charts/global_deaths_choropleth_map.html %}

## U.S. deaths Choropleth map

{% include covid-charts/us_deaths_choropleth_map.html %}

# Population-based statistics

## Active cases per million people

{% include covid-charts/active_cases_per_million_people_bar_chart.html %}

## Deaths per million people

{% include covid-charts/deaths_per_million_people_bar_chart.html %}

# Case fatality rate

{% include covid-charts/most_affected_countries_fatality_rate_bar_chart.html %}

# Flatten the curve

{% include covid-charts/most_affected_countries_daily_deaths_bar_chart.html %}

# Link between aging and virus fatality

{% include covid-charts/age_vs_fatality_rate_scatter_plot.html %}