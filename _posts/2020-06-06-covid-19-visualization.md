---
title: COVID-19 Interactive Visualization
layout: single
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



{% include covid-charts/most_affected_countries_pie_chart.html %}

## Most affected countries bar chart

{% include covid-charts/most_affected_countries_stacked_bar_chart.html %}

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