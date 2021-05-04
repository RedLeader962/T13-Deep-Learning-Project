from setuptools import setup

setup(
    name='team13twin-rudder',
    version='1.1',
    packages=['ppo', 'logger', 'rudder', 'tests', 'tests.experiment', 'tests.test_codebase', 'script', 'codebase',
              'codebase.ppo', 'codebase.logger', 'codebase.rudder', 'experiment', 'experiment_runner'],
    package_dir={'': 'codebase'},
    author='Luc Coupal, Francois-Alexandre Tremblay, William-Ricardo Bonilla-Villatoro)',
    author_email='luc.coupal.1@ulaval.ca,francois-alexandre.tremblay.1@ulaval.ca,william-ricardo.bonilla-villatoro.1@ulaval.ca',
    install_requires=[
        'pytest',
        'gym',
        'matplotlib',
        'numpy',
        'pandas',
        'torch',
        'ipykernel',
        'scikit-learn',
        'ipython',
        'scipy',
        ],
    )
