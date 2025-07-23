from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(filename:str) -> List[str]:
    try:
        with open(filename) as f:
            requirements = f.read().splitlines()

            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)

        return requirements
    except FileNotFoundError:
        return []
    

setup(
    name= 'End-to-End-MLProject',
    version= '0.0.1',
    author= 'Akhila',
    author_email= 'devarapalliakhila@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)