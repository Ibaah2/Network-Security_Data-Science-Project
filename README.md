This is an end to end data science network security project

## The steps are as follows:

First, we create an environment using the cmd command prompt

conda create -p venv python==3.10

If conda is not working, make sure the path for conda is added to the system variable

First, search the location of where anaconda is installed Eg: C:\ProgramData\anaconda3
Make sure "Scripts" and "Library" exists.

Go to windows (Win + S) and search for environment variable.
Click on environment variable.
Under the system varibles, add the new paths:
C:\ProgramData\anaconda3\Library\bin
C:\ProgramData\anaconda3\Scripts
C:\ProgramData\anaconda3\condabin

Check your computer to ensure the right paths

Restart VS code and run "where conda" or "conda --version" in the command promt to ensure conda exists

If conda is still not working, make sure to open the anaconda prompt on your local computer, then change the working directory to the existing folder you are working in.

Restart VS code and run those commands

## Now, lets create the environmet
conda create -p venv python==3.10

Activate the environment
conda activate C:\Users\user 1\Documents\My_Learning\Personal_Projects\Network_Security\venv

Deactivate the environment
conda deactivate


## Logging and Exception Handling

### Logging
This project includes a structured logging setup to track events and support debugging throughout the data pipeline. Each run of the project generates a timestamped log file stored in a dedicated logs/ directory.

Key Features
Automatically creates a separate log file for every execution
Organizes logs in a central logs/ folder
Includes useful details in each log entry, such as:
-Timestamp
-Line number
-Module or script name
-Log level (INFO, WARNING, ERROR, etc.), INFO was used in this project
-Descriptive message

### Exception handling
This project includes a custom exception class designed to capture detailed error information during execution. It is fully integrated with the logging system.

Tracks exactly where and why an error occurred

Captures:
- File name
- Line number
- Error message

Logs all exception-related events for better debugging and analysis

