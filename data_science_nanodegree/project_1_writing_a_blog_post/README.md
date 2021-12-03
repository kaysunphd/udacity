## Purpose

### The publicly available [StackOverflow survey data](https://insights.stackoverflow.com/survey) of 65,000 developers worldwide was analyzed using the [cross-industry standard process for data mining (CRISP-DM)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) approach.

### CRISP-DM consists of 6 phases:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment


### 1. Business Understanding
#### The goal of the analysis is to understand how to become a software developer by answering 3 basic questions:

#### 1) There are many different languages, web framework, databases, platforms, and collaborative tools being used. It would be challenging to learn all of them, so which ones are the most useful to learn first based on popularity to get started?

#### 2) Expertise in certain in-demand software languages can command high salaries. Which are the higher paying software languages and how many years of experience are needed?

#### 3) While software development should be democratized to everyone and anyone, but the realities of societal prejudices and inequalities bias one group of people over others. What are the ethnicity and gender distributions among current developers?


### 2. Data Understanding
##### The survey data consists of 64,461 respondents (or rows) and 61 survey questions (or columns). Since all the columns are not needed to answer the 3 main questions above, focus is placed on the following ones:

Column name| Description
-----|-----
ConvertedComp| Annual compensation in USD
DatabaseWorkedWith| Databases respondent is working with
Ethnicity | Social group respondent identifies with
Gender | Gender respondent identifies with
LanguagesWorkedWith | Languages respondent is working with
MiscTechWorkedWith | Miscellaneous technology respondent is working with
NewCollabToolsWorkedWith | Collaborative tools respondent is working with
PlatformWorkedWith | Platform respondent is working with
WebFrameWorkedWith | Web framework respondent is working with
YearsCodePro | Years coding professionally

### 3. Data Preparation
#### To preserve as many valid survey respondents as possible, missing values only to columns of interest for specific questions were dropped rather than across the entire board or dataframe. 
#### The data was read in as String for all entries. Variables that were numeric, like ConvertedComp and YearsCodePro, were converted to Float.

#### File Descriptions
- [stackoverflow_insights_2020.ipynb](https://github.com/kaysunphd/blog/blob/main/stackoverflow_insights_2020.ipynb) is the Jupyter notebook used for analysis.
- Download the [source data](https://drive.google.com/file/d/1dfGerWeWkcyQ9GX9x20rdSGj7WtEpzBB/view) and place the developer_survey_2020 folder in the same path as the notebook.

#### Libraries required to run
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org)
- [pandas](https://pandas.pydata.org/)

#### The main takeaways are:
##### 1) Popular software tools in 2020 were focused on web applications using JavaScript (jQuery, React.js), HTML/CSS and SQL (MySQL) developed on either Linux and Windows platforms.
##### 2) In 2020, highest paid developers were for Scala and React despite their lower popularity.
##### 3) Non-White and female developers were still severely underrepresented in the field in 2020.

### 4 to 6. Modeling, Evaluation and Deployment
#### To answer the questions posed, only a direct description analysis is needed to observe the trends and statistics. For any future predictive analyses, more complex machine learning methods would be needed.

### Read more at [blog](https://medium.com/@kaysun_37703/who-wants-to-be-a-developer-722b06cd22ea).

### Acknowledgements
- StackOverflow for the [survey data](https://insights.stackoverflow.com/survey)
