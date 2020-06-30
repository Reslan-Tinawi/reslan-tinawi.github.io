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

# What is this post about

In this post I'll present different visuals of the novel *Coronavirus (COVID-19)*, to show how it impacted the whole world disproportionately, and how contagious it is.

I'll use dataset from the [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) which contains daily aggregated data, and time series data.

The data provides us with the following four variables:

- **Active**: the number of active cases, cases which have tested positive for the virus.

- **Recovered**: number of recovered cases from the virus.

- **Deaths**: number of deaths.

- **Confirmed**: the sum of the previous the three variables.

These variables are provided for each country on a daily basis.

I'll also use two data sources from the [World Bank Group](https://www.worldbank.org/):

- [Total population](https://data.worldbank.org/indicator/SP.POP.TOTL).

- [Population ages 65 and above as a percentage of the total population](https://data.worldbank.org/indicator/SP.POP.65UP.TO.ZS).

The population data might not be very accurate, as it was last updated in 2018, but that won't be a problem.

All the visualizations were created using the amazing library: [Plotly](https://plotly.com/python/).

The code for data analysis and visualization can be found [here](https://github.com/Reslan-Tinawi/COVID-19-Visualization).

*Note*: this post is not intended to be a real-time dashboard of the COVID-19 cases, therefore it won't be updated anymore, the data used here is from 22/01/2020 till 30/06/2020.

# Overall statistics pie chart

The following pie chart shows the percentages of the three measures: active cases, recovered cases, and total deaths.

At first sight, this chart comforts us! the virus doesn't seem to be very *fatal*, only about 5% of the total cases are **deaths**, while 51% of the cases had **recovered** from the virus.

{% include covid-charts/overall_stats_pie_chart.html %}

Pie Chart is misleading in interpreting the severity level of the virus, it doesn't show us how fast the virus is spreading, and how the number of deaths is growing *exponentially*.

Later in this post, I'll use different type of charts to answer the previous questions.

# Treemap chart

The following treemap describes the different proportions countries around the world have been affected by this pandemic, grouped together by continent.

{% include covid-charts/most_affected_countries_treemap_chart.html %}

This treemap serves as an overall-look at the total deaths in the most affected countries.

We can see that **Europe** and **North America** constitute around half of the global deaths.

# Most hit countries

After taking a general overview on the virus statistics, and how it has taken down many lives across the world, here in this section I'll address the most hit countries in terms of their death tolls.

## Most affected countries pie chart

Looking at the following chart, we can notice that the **U.S.** and the **European** countries are among the most hit countries, and together they constitute *almost* the whole chart.

{% include covid-charts/most_affected_countries_pie_chart.html %}

We can also observe that the chart is divided into three (nearly) equal parts:

- The U.S.

- Brazil, The UK, and Italy.

- And the remaining eleven countries.

Based on the previous observation, we might ask why the U.S. death tolls are so high (as much as eleven countries combined)?

The U.S. is the third in population size (327 million people as of 2018), so we can't compare it with other low-populated countries.

Later in this post, I'll introduce another measures based on the countries populations, to look at the most hit countries from a different perspective.

## Most affected countries bar chart

The previous chart showed us only the death tolls, and discarded the other two variables: active cases, and recovery cases.

The chart below explores the three measures for each country.

{% include covid-charts/most_affected_countries_stacked_bar_chart.html %}

This chart helps us understand the relative proportion of each variable compared to the total confirmed cases in a particular country.

One country that stands out among others is **Germany**, while it has as much confirmed cases as other european countries (France, Italy, and Spain), it managed to control the number of active cases, and ended up with the highest recovery rate, mainly because of its excessive testing.

More on Germany's response to the pandemic in this article: [A German Exception? Why the Country’s Coronavirus Death Rate Is Low](https://www.nytimes.com/2020/04/04/world/europe/germany-coronavirus-death-rate.html)

*Note*: Both The UK and Netherlands have very low number of recovery cases, that the recovery cases in these countries are not visible in the chart, but the recoveries are estimates, so they are not accurate, quoting the dataset documentation of how the recovery cases are computed:

> Recovered cases outside China are estimates based on local media reports, and state and local reporting when available, and therefore may be substantially lower than the true number.

# How contagious is the virus

All the previous charts presented the aggregated statistics of the pandemic, ignoring the growth of the virus over time, and giving us no answer of how fast the virus is spreading, and how the number of deaths is increasing rapidly?

In this, and in the following section I'll take the *time variable* into account, to look at this pandemic from a different angle.

The following three charts illustrate the accumulated sum of active cases, deaths, and recovered cases over time.

## Cases over time

{% include covid-charts/active_cases_line_chart.html %}

## Deaths over time

{% include covid-charts/death_tolls_line_chart.html %}

## Recovery over time

{% include covid-charts/recovered_cases_line_chart.html %}

At first look, we can observe that the death tolls are growing exponentially! and we can notice some other observations:

- The number of infections in the U.S. has skyrocketed in no time.

- While most countries have reached *more or less* their peak value in terms of death tolls, Brazil and Mexico are still far from reaching theirs.

- Italy, Spain, France and the UK were almost identical in their death tolls, until the beginning of May, then the UK surpassed Italy in the recorded deaths, and became Europe's COVID-19 epicenter.

# Choropleth maps

A choropleth map is a type of thematic map in which areas are colored relatively to a statistical variable that corresponds to a geographic characteristic within each area.

Here, I'll create two choropleth maps for depicting the death tolls globally and in the U.S. alone.

In the previous *line charts* only a handful set of countries were shown, so the lines don't get cluttered and the figure doesn't lose its quality, using choropleth maps we can show the growth of death tolls for all the countries.

Another advantage of choropleth maps is that they provide geographical information, which can be useful to investigate clustering patterns of the virus.

*Note*: use the `play` button to start the animation of the map, the animation frame corresponds to a single day.

## Global deaths Choropleth map

{% include covid-charts/global_deaths_choropleth_map.html %}

The virus started in China, and it remained the epicenter of the epidemic until late-March, by that time a cluster of European countries have emerged consisting of Italy, Spain and France, which were reporting the highest death tolls outside China, and Italy became the new epicenter.

Over the next few days, the cluster expanded more and more, to include more neighboring countries, among these countries: the UK, Netherlands, Germany, and Belgium.

After passing the first 10 days of April, the U.S. death tolls have surpassed Italy's, to become the new epicenter.

By mid-June Brazil's death tolls surpassed all the other European countries, and it became the second deadliest country, after the U.S.

## U.S. deaths Choropleth map

{% include covid-charts/us_deaths_choropleth_map.html %}

Looking closer at the U.S. deaths (over time) at the state-level, we see that the first deaths were reported in Washington in early-March.

By late-March, different states were reporting deaths, and *New York* was on the lead, with nearly 2,500 deaths.

New York remained the most-hit state in the U.S., and by late-May, a cluster of neighboring states with the highest death tolls have emerged, consisting of: New York, New Jersey, Massachusetts, Pennsylvania and Connecticut.

This 5-states cluster reported the highest death tolls among other states, with nearly 65 thousand deaths, which is more than *half* of the U.S. total death tolls.

It's clear that the pandemic has affected the U.S. states disproportionally, some states reported very high death tolls (in terms of thousands, and ten thousand), while others were barely affected by this pandemic, and reported only a few deaths.

# Population-based statistics

So far different types of charts were employed to study the data, and answer different questions, but they all treated different countries the same.

To understand how this pandemic has impacted on different countries, it's not enough to compare countries by their death tolls, for example, **Germany** and **Belgium** have almost equal death tolls, but their population sizes are relatively different, where Germany's population size is equal to almost 7 times of Belgium's population, which means that Belgium has a *much worse* situation compared to Germany.

An important data that has been ignored so far is the population size data in each country, which can be used to show the relative number of deaths / active cases compared to the country's population size.

Here, in this section, I'll use population size data, and introduce two new measures:

- Active cases per million people: this variable describes the number of active cases per one million people.

- Deaths per million people: this variable describes the number of deaths per one million people.

## Active cases per million people

{% include covid-charts/active_cases_per_million_people_bar_chart.html %}

Although the U.S. has a much higher number of active cases than the UK, we can see that they have very close cases per million people, that's mainly because UK's population is very small in comparison with the U.S.'s population.

## Deaths per million people

{% include covid-charts/deaths_per_million_people_bar_chart.html %}

Interestingly, this chart reveals completely new information.

Belgium has the highest deaths per million people rate, with almost one thousand deaths per million people.

The UK, Spain, Italy, France, and the Netherlands also have very high deaths per million people rate, which suggests that these European countries have lost many lives, although some of them don't have a very high death toll.

India is a high populated country, that's why it has very low deaths per million rate.

# Case fatality rate

How deadly is this virus? and how likely is it that a person dies from this virus, after getting infected?

In [Epidemiology](https://en.wikipedia.org/wiki/Epidemiology) (the science of studying diseases), a [case fatality rate (CFR)](https://en.wikipedia.org/wiki/Case_fatality_rate) is the proportion of people who die from a certain disease among all individuals diagnosed with the disease, and it's typically used as a measure of disease severity.

Here, I'll use the following equation to calculate the case fatality rate:

$$ Case\ Fatality\ Rate\ (CFR) = \frac{total\ deaths \ast 100}{total\ confirmed\ cases}$$

{% include covid-charts/most_affected_countries_fatality_rate_bar_chart.html %}

We can see that Belgium has the highest fatality rate with 16% of confirmed cases are deaths, other countries that have *relatively* high fatality rate (more than 10%) are: France, Italy, the UK, Netherlands, Mexico and Spain.

The fatality rate can be linked to many factors, like:

- The quality of healthcare.

- The average age of the population.

Europe is known for its high percentage of elderly people, this might be related to the high fatality rates in this continent, the final section of this post is dedicated to study the association between aging and fatality rate.

# Flatten the curve

Flattening the curve involves reducing the number of new COVID-19 cases from one day to the next. This helps prevent healthcare systems from becoming overwhelmed.

When a country has fewer new COVID-19 cases emerging today than it did on a previous day, that’s a sign that the country is flattening the curve.

This chart illustrates the daily new deaths in the most hit countries.

*Note*: in the upper-left of this figure, there's a dropdown menu, which you can use it to switch between countries.

{% include covid-charts/most_affected_countries_daily_deaths_bar_chart.html %}

Most countries have reached their peak value (in terms of daily deaths) in the time between late-March and early-April. The U.S. took a bit longer to reach the peak point, until the end of April.

Brazil and Mexico, on the other hand, are recording increasing new deaths each day, and they are still far from reaching their peak values.

Iran is going through what is called a *second wave*, the daily deaths kept decreasing until the beginning of June, after that the daily deaths started to increase again, which may *suggests* that there's a second wave of high deaths in this country.

# The relation between aging and virus fatality

In this section I'll study the relation between the fatality rate and aging, which is important to address the question:

> Do countries with higher percentages of the elderly have higher fatality rates?

The following chart outlines the relationship between the percentage of elderly (people whose age is over 65 years) and the fatality rate, for 40 different countries.

The size of each point in this plot is relative to the country's total death tolls.

{% include covid-charts/age_vs_fatality_rate_scatter_plot.html %}

We can see that the relation is not *quite* linear, the line of best fit (an ordinary least squares regression line) captures the linear relation between these two variables, and how the points should be scattered.

The points are not spread near the line, so the relation is not linear, which means that the fatality rate is affected by some other factors, like the quality of health care, country's GDP (Gross domestic product).

From this chart, we can notice the following observations:

- **Japan** has the highest percentage of elderly, but its fatality rate is very low, and its death toll is relatively low.

- Both **Mexico** and **Ecuador** have low percentages of elderly (only 7%), however, they have a high fatality rate. This high percentage of fatality might be related to the fact that these countries are poor and don't have a well-prepared health system for dealing with this pandemic.

- Most European countries have a high percentage of elderly (between 16% and 22%), but in terms of fatality rate, they cluster into two groups:

  - High fatality rate countries: Belgium, the UK, France, Spain and Italy all have a fatality rate higher than 10%.

  - Low fatality rate countries: Switzerland, Germany and Austria have a much lower fatality rate (lower than 7%) compared with their neighboring countries.
