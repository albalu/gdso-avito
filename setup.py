#!/usr/bin/env python

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='gdso-avito',
        version='0.1.0',
        description='Avito Kaggle GDSO data science workshop',
        url='https://github.com/albalu/gdso-avito',
        author='Alireza Faghaninia, Zexuan Xu, Winnie Lee, Darius Parvin',
        author_email='alireza.faghaninia@gmail.com',
        license='modified BSD',
        packages=find_packages(),
        package_data={},
        zip_safe=False,
        install_requires=[],
        extras_require={},
        classifiers=['Programming Language :: Python :: 3.6',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Data Scientists',
                     'Intended Audience :: Students',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     ],
        test_suite='nose.collector',
        tests_require=['nose'],
        scripts=[]
    )
