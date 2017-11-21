# Copyright 2017,2018 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Setup script for ``dipp``.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

import os
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand
import sys


root_path = os.path.dirname(__file__)


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


test_path = os.path.join(root_path, 'dipp', 'test')


def find_tests():
    """Discover the test files for packaging."""
    tests = []
    for path, _, filenames in os.walk(os.path.join(root_path, test_path)):
        for filename in filenames:
            basename, suffix = os.path.splitext(filename)
            if (suffix == '.py' and
                    (basename.startswith('test_') or
                     basename.endswith('_test'))):
                tests.append(os.path.join(path, filename))

    return tests


# Determine version from top-level package __init__.py file
with open(os.path.join(root_path, 'dipp', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

description = ('Deep Inverse Problems Package - '
               'a library of add-ons to various Deep Learning frameworks for '
               'solving Inverse Problems')

long_description = 'TODO'

setup(
    name='dipp',

    version=version,

    description=description,
    long_description=long_description,

    url='https://github.com/odlgroup/dipp',

    author='Jonas Adler, Holger Kohr',
    author_email='odl@math.kth.se',

    license='MPL-2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers

    keywords='research development mathematics prototyping deep-learning',

    packages=find_packages(),
    package_dir={'dipp': 'dipp'},

    package_data={'dipp': find_tests() + ['dipp/pytest.ini']},
    include_package_data=True,

    install_requires=[],
    tests_require=['pytest', 'pytest-pep8'],
    extras_require={
        'pytorch': ['pytorch'],
        'tensorflow': ['tensorflow'],
        'theano': ['theano']
    },

    cmdclass={'test': PyTest},
)
