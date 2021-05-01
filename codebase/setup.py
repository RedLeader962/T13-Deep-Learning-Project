from setuptools import setup

setup(
    name='team13twin-rudder',
    version='',
    packages=['ppo', 'logger', 'rudder', 'tests', 'tests.experiment', 'tests.test_codebase', 'script', 'codebase',
              'codebase.ppo', 'codebase.logger', 'codebase.rudder', 'experiment', 'experiment_runner'],
    package_dir={'': 'codebase'},
    url='',
    license='',
    author='',
    author_email='',
    description='',
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
