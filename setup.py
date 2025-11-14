from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='openmmslicer',
    version='3.1.0',
    packages=['openmmslicer'],
    install_requires=requirements,
    url='',
    license='GPL',
    author='Miroslav Suruzhon, Justina Ratkeviciute',
    author_email='jr1u18@soton.ac.uk',
    description='Sequential LIgand Conformation ExploreR (SLICER)---A Sequential Monte Carlo Sampler for OpenMM'
)
