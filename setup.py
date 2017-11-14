#!/usr/bin/env python2

from setuptools import setup

if __name__ == '__main__':
    setup(name='dapy',
          description='Data assimilation in Python',
          author='Matt Graham',
          url='https://github.com/matt-graham/data-assimilation.git',
          packages=['dapy', 'dapy.inference', 'dapy.models'])
