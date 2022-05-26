#!/usr/bin/env python

from distutils.core import setup

setup(
    name='SchemaMechanism',
    version='0.1.0',
    description='A Python version of Drescher\'s Schema Mechanism',
    author='Sean Kugele',
    author_email='seankugele@gmail.com',
    url='https://github.com/skugele/schema-mechanism',
    license='MIT',
    python_requires='>=3.9',
    install_requires=[
        'anytree==2.8.0',
        'lark==1.1.1',
        'numpy==1.21.4',
        'pynput==1.7.6'
        'scikit-learn==1.0.1',
        'scipy==1.7.3',
    ],
)
