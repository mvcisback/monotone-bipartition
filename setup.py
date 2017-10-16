from setuptools import setup, find_packages

setup(name='multidim-threshold',
      version='0.1',
      description='TODO',
      url='http://github.com/mvcisback/multidim-threshold',
      author='Marcell Vazquez-Chanlatte',
      author_email='marcell.vc@eecs.berkeley.edu',
      license='MIT',
      install_requires=[
          'numpy',
          'scipy',
          'funcy',
      ],
      packages=find_packages(),
      )
