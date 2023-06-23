from setuptools import setup, find_packages

setup(
    name='robust_codesign',
    author='Federico Girlanda',
    version='2.0.0',
    url="https://git.hb.dfki.de/underactuated-robotics/robust_codesign",
    packages=find_packages(),
    install_requires=[
        # general
        'numpy',
        'matplotlib',
        'scipy',
        'ipykernel',
        'pyyaml',
        'pandas',
        'argparse',
        'sympy',
        'lxml',

        # optimal control
        #'drake==1.5.0',
        'filterpy',
        'cma'
    ],
    classifiers=[
          'Development Status :: 5 - Stable',
          'Environment :: Console',
          'Intended Audience :: Academic Usage',
          'Programming Language :: Python',
          ],
)