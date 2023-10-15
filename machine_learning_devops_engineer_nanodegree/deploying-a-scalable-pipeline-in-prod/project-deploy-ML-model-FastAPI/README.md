# Environment Set up
## Local machine
* Use the supplied requirements file to create a new environment, or
* conda create -n [envname] "python=3.11.5" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
* Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories
* Setup GitHub Actions on your repo. Make sure to have the same version of Python as you used in development.

# Data
* Download [census.csv](https://archive.ics.uci.edu/dataset/20/census+income).
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.


# To run
in VSCode, update settings.json to <br>
{
    "files.autoSave": "afterDelay",
    "terminal.integrated.env.osx": { "PYTHONPATH": "${workspaceFolder}" }
}

mlflow run src/basic_cleaning -P output_artifact="cleaned_census.csv" -P output_type="clean_sample" -P output_description="Cleaned data"
<br>
mlflow run src/ml -P input_artifact="cleaned_census.csv"
<br>
mlflow run src/test_runs

# To Render
Environment: PYTHON_VERSION | 3.11.5
Buikd Command: pip install --upgrade pip -r requirements.txt
