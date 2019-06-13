from setuptools import setup, find_packages
from os import path

from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = ['numba','numpy','opencv-python']

setup(
    name='img_term',  # Required
    version='1.0.0',  # Required
    description='A small script to render images to an ANSI terminal',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/jonnoFTW/img_term',  # Optional
    author='Jonathan Mackenzie',
    classifiers=[  # Optional
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='terminal ansi fun',  # Optional
    packages=['img_term'],  # Required
    install_requires=requirements,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={  # Optional
        'console_scripts': [
            'img_term = img_term.img_term:main',
        ],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jonnoFTW/img_term/issues',
        'Source': 'https://github.com/jonnoFTW/img_term/',
    },
)
