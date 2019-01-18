from setuptools import find_packages, setup


setup(
    name='monotone-bipartition',
    version='0.0.0',
    description='Compute Monotone Threshold Surfaces and compute distances between surfaces.',
    url='http://github.com/mvcisback/monotone-bipartition',
    author='Marcell Vazquez-Chanlatte',
    author_email='marcell.vc@eecs.berkeley.edu',
    license='MIT',
    install_requires=[
        'funcy',
        'lazytree',
        'lenses',
        'numpy',
    ],
    packages=find_packages(),
)
