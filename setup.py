'''
The setup.py file is an essential part of packaging and distributing python projects. It is used by setuptools 
(or distutils in older python versions) to define the configuration of projects, such as metadata, dependencies
and more.

Supposed your are building a robot, you will need some instruction booklets and a toolbox to help you out.
- In python, the project we are trying to build "Network Security using Phishing data" is the "robot"
- The packages we will be building in "NetworkSecurity" folder and subfolders "components", "constants", etc., 
  are the boxes that contain the parts of the robot we need to execute the project.
- The set of instructions needed is contained in "setup.py"
'''

from setuptools import find_packages, setup
from typing import List

def get_requirements() ->List[str]:
    '''
    This function will return a list of requirements
    '''

    requirement_list = []

    try:
        with open('requirements.txt', 'r') as file:
            #Read lines from file
            lines = file.readlines()
            #Process each line in lines
            for line in lines:
                requirement = line.strip() ## Remove any empty spaces
                if requirement and requirement!= '-e .': ## Ignore any -e.
                    # Add requirements to the empty list 'requirement_list'
                    requirement_list.append(requirement) 
    except FileNotFoundError:
        print('requirements.txt file not found')

    return requirement_list

#print(get_requirements())

setup(
    name='NETWORK_SECURITY',
    version='0.1',
    author='Isaac Baah',
    author_email='baahisaac18@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
    )