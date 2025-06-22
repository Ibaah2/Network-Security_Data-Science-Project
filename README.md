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