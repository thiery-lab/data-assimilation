"""Set up script for Python data assimilation package."""

from setuptools import setup, Extension
import argparse
import sys

parser = argparse.ArgumentParser(
    description='Python data assimilation package setup')
parser.add_argument('-debug', action='store_true', default=False,
                    help='Export GDB debug information when compiling.')
parser.add_argument('-use-cython', action='store_true', default=False,
                    help='Use Cython to compile from .pyx files.')
parser.add_argument('-use-gcc-opts', action='store_true', default=False,
                    help='Use GCC compiler optimisations for quicker but '
                         'less safe mathematical operations.')
parser.add_argument('-use-cython-opts', action='store_true', default=False,
                    help='Add extra Cython compile directives for quicker '
                         'but less safe array access. Requires -use-cython '
                         'to be set to have an effect.')

# hack to get both argparser help and setuptools help displaying
if '-h' in sys.argv:
    sys.argv.remove('-h')
    help_flag = '-h'
elif '--help' in sys.argv:
    sys.argv.remove('--help')
    help_flag = '--help'
else:
    help_flag = None

args, unknown = parser.parse_known_args()

# remove custom arguments from sys.argv to avoid conflicts with setuptools
for action in parser._actions:
    for opt_str in action.option_strings:
        try:
            sys.argv.remove(opt_str)
        except ValueError:
            pass
# if a help flag was found print parser help string then read so setuptools
# help also displayed
if help_flag:
    parser.print_help()
    sys.argv.append(help_flag)

ext = '.pyx' if args.use_cython else '.c'

extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']

if args.use_gcc_opts:
    extra_compile_args += ['-O3', '-ffast-math']


if args.use_cython and args.use_cython_opts:
    compiler_directives = {
        'boundscheck': False,  # don't check for invalid indexing
        'wraparound': False,  # assume no negative indexing
        'cdivision': True,  # don't check for zero division
        'initializedcheck': False,  # don't check memory view init
        'embedsignature': True  # include call signature in docstrings
    }
else:
    compiler_directives = {}

ext_modules = [
    Extension('dapy.models.integrators',
              ['dapy/models/integrators' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('dapy.models.lorenz63integrator',
              ['dapy/models/lorenz63integrator' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('dapy.models.Lorenz96Integrator6integrator',
              ['dapy/models/lorenz96integrator' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args)
]

if args.use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules,
                            compiler_directives=compiler_directives,
                            gdb_debug=args.debug)

packages = [
    'dapy',
    'dapy.models',
    'dapy.inference'
]

if __name__ == '__main__':
    setup(
        name='dapy',
        description='Data assimilation in Python',
        author='Matt Graham',
        url='https://github.com/matt-graham/data-assimilation.git',
        packages=packages,
        ext_modules=ext_modules,
    )
