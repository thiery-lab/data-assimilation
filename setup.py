"""Set up script for Python data assimilation package."""

from setuptools import setup, Extension
import argparse
import sys

parser = argparse.ArgumentParser(
    description='Python data assimilation package setup')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Export GDB debug information when compiling.')
parser.add_argument('--use-cython', action='store_true', default=False,
                    help='Use Cython to compile from .pyx files.')
parser.add_argument('--use-gcc-opts', action='store_true', default=False,
                    help='Use GCC compiler optimisations for quicker but '
                         'less safe mathematical operations.')
parser.add_argument('--no-cython-opts', action='store_true', default=False,
                    help='Remove extra Cython compiler directives for quicker '
                         'but less safe array access. Requires --use-cython '
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
# if a help flag was found print parser help string then readd so setuptools
# help also displayed
if help_flag:
    parser.print_help()
    sys.argv.append(help_flag)

c_ext = '.pyx' if args.use_cython else '.c'
cpp_ext = '.pyx' if args.use_cython else '.cpp'

extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']

if args.use_gcc_opts:
    extra_compile_args += ['-O3', '-ffast-math']


if args.use_cython and args.use_cython_opts:
    compiler_directives = {
        'boundscheck': False,  # don't check for invalid indexing
        'wraparound': False,  # assume no negative indexing
        'nonecheck': False,  # don't check for field access on None variables
        'cdivision': True,  # don't check for zero division
        'initializedcheck': False,  # don't check memory view init
        'embedsignature': True,  # include call signature in docstrings
        'language_level': 3,  # Python 3
    }
else:
    compiler_directives = {}

ext_modules = [
    Extension('dapy.integrators.implicitmidpoint',
              ['dapy/integrators/implicitmidpoint' + c_ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('dapy.integrators.lorenz63',
              ['dapy/integrators/lorenz63' + c_ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('dapy.integrators.lorenz96',
              ['dapy/integrators/lorenz96' + c_ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('dapy.integrators.interpolate',
              ['dapy/integrators/interpolate' + c_ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('dapy.ot.solvers',
              ['dapy/ot/solvers' + cpp_ext],
              language='c++', include_dirs=['dapy/ot'],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
]

if args.use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules,
                            compiler_directives=compiler_directives,
                            gdb_debug=args.debug)

packages = [
    'dapy',
    'dapy.inference',
    'dapy.integrators',
    'dapy.models',
    'dapy.ot',
    'dapy.utils'
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
