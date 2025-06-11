Repository for the ICRA2025 Workshop Paper "Seeing, Saying, Solving" 
# Instructions:
## Setup Requirements
### Environment setup
* We recommend using Conda or venv.

* Conda config with dependencies are listed in the s3_conda_config.yml.

* With Conda is installed, create a new conda environment `s3_conda` by:

  * `conda env create -f s3_conda_config.yml`

### Obtain Gurobi academic license
* Our code uses Gurobi optimizer as the primary MIP solver.
* You do not have to install nor obtain the named academic license for our code to work (although it is free!)
  * Obtain Gurobi WLS academic license (also free!) - https://www.gurobi.com/features/academic-wls-license/
  * Follow the instructions until you can download the `gurobi.lic` license file locally on your computer.
* Tips for those running on Ubuntu:
  * You can set the enviroment variable: `export GRB_LICENSE_FILE=/PATH_TO_LICENSE_FILE/gurobi.lic`
    * e.g) `export GRB_LICENSE_FILE=~/Downloads/gurobi.lic`
  * We recommend adding this line to bashrc by: `echo 'export GRB_LICENSE_FILE=/PATH_TO_LICENSE_FILE/gurobi.lic' >> ~/.bashrc`  
### Generate Openrouter API Key
* Obtain a free Openrouter API Key here: https://openrouter.ai/
  * We default to deepseek model in the code, but with openrouter API, you can use any model (e.g gpt-4).
* Once the API key is generated, create a .env file inside the root repo:
  * `touch .env`
  * `nano .env`
  * Inside the .env:
    * OPENROUTER_API_KEY='YOUR KEY HERE'
* Don't forget to add the .env to your .gitignore

## Running Experiment 1:


