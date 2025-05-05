# Contrail_Forecast : How to get a new inference

git clone https://github.com/Mariaccia-Robin/Contrail_Forecast.git
python -m venv venv

#if system does not allow to run scripts : Set-ExecutionPolicy RemoteSigned
venv/Scripts/activate
pip install -r Contrail_Forecast/requirements.txt


Then you must also copy the two .GRIB files I delivered with my project inside the data/ folder
(They were too big to commit with GIT)


python Contrail_Forecast path_to_input_file path_you_want_output_file_to_be
#note : If you use relative paths, the project's working directory is the root directory of the project


The output file will be a copy of your input file with a new column "Forecasted Contrails (kgCO2e)"