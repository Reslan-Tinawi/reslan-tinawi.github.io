---
title: COVID-19 Interactive Visualization
layout: single
classes: wide
toc: true
custom_css: plotly
---

**Last Updated: 28/05/2020**

This post will demonstrates different interactive visualizations of the novel *Cororna Virus* (COVID-19). The dataset used is provided by [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19), and the visualization were made using [Plotly Python](https://plotly.com/python/). Plotly is super-easy, full packed, and *interactive* visualization library for `Python`.

# Virus overall stats:
{% include covid-charts/overall_stats_pie_chart.html %}

# Most affected countries:
The following pie chart, shows the first ten countries, by the number of deaths.

{% include covid-charts/most_affected_countries_pie_chart.html %}

From this graph, we can see that the most hit countries by the pandemic, are the european countries, and the U.S., although the virus originated from China.

Italy, Spain, and France death tolls constitute half of the chart, while the U.S. consitutes a quarter.

It's noticable that China has a very low death toll, compared to the other countries, although the viruse originated there, and it was the epicenter of the epidemic.

First confirmed cases date:

<div class="table-container" style="width: 350px; height: 300px;">
    <iframe src="/assets/visualizations/first_infection_date_table.html" id="igraph" scrolling="no" style="border: none; position: relative; height: 100%; width: 100%;" seamless="seamless"></iframe>
</div>

First deaths date:

<div class="table-container" style="width: 350px; height: 300px;">
    <iframe src="/assets/visualizations/first_death_date_table.html" id="igraph" scrolling="no" style="border: none; position: relative; height: 100%; width: 100%;" seamless="seamless"></iframe>
</div>

As it's shown in these tables, it took time for the viruse to spread across the countries, while it's was only living in China.


For example, Italy and the U.S. had the first deaths reported in the late of February, which is a month later after the first deaths were reported in China, and yet somehow *tragically*, they ended up with the highest death tolls in the world.

# Glance over time
Since the data provides a date of the observed measures on a daily basis, I will try to show how the virus spread progressively, and how the number of cases growing exponentially.

Deaths Trajectory
{% include covid-charts/deaths_trajectory_line_chart.html %}

Infections Trajectory
{% include covid-charts/infections_trajectory_line_chart.html %}

Recovered Trajectory
{% include covid-charts/recovery_trajectory_line_chart.html %}

Choropleth map

{% include covid-charts/global_deaths_choropleth_map.html %}

{% include covid-charts/us_deaths_choropleth_map.html %}

Most affected countries charts (stacked bar char, horizontal stacked bar chart, treemap)

{% include covid-charts/most_affected_countries_stacked_bar_chart.html %}

{% include covid-charts/most_affected_countries_horizontal_stacked_bar_chart.html %}

{% include covid-charts/most_affected_countries_treemap_chart.html %}

Countries fatalities rate:

{% include covid-charts/most_affected_countries_fatality_rate_bar_chart.html %}

Cases per million people:

{% include covid-charts/active_cases_per_million_people_bar_chart.html %}

{% include covid-charts/deaths_per_million_people_bar_chart.html %}

Aging and fatality rate relation:

{% include covid-charts/age_vs_fatality_rate_scatter_plot.html %}