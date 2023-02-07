"""Setup script for building Cython extension modulesefault c or c++."""

from setuptools import setup, Extension
import argparse
import pathlib
import sys
import numpy

parser = argparse.ArgumentParser(
    description="Python data assimilation package setup"
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Export GDB debug information when compiling."
)
parser.add_argument(
    "--use-cython",
    action="store_true",
    default=False,
    help="Use Cython to compile from .pyx files."
)
parser.add_argument(
    "--use-gcc-opts",
    action="store_true",
    default=False,
    help="Use GCC optimisations for quicker but less safe mathematical operations."
)
parser.add_argument(
    "--no-cython-opts",
    action="store_true",
    default=False,
    help=(
        "Remove extra Cython compiler directives for quicker but less safe array "
        "access. Requires --use-cython to be set to have an effect."
    )
)

# hack to get both argparser help and setuptools help displaying
if "-h" in sys.argv:
    sys.argv.remove("-h")
    help_flag = "-h"
elif "--help" in sys.argv:
    sys.argv.remove("--help")
    help_flag = "--help"
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

extra_compile_args = ["-fopenmp"]
extra_link_args = ["-fopenmp"]

if args.use_gcc_opts:
    extra_compile_args += ["-O3", "-ffast-math"]

if args.use_cython and not args.no_cython_opts:
    compiler_directives = {
        "boundscheck": False,  # don't check for invalid indexing
        "wraparound": False,  # assume no negative indexing
        "nonecheck": False,  # don't check for field access on None variables
        "cdivision": True,  # don't check for zero division
        "initializedcheck": False,  # don't check memory view init
        "embedsignature": True,  # include call signature in docstrings
        "language_level": 3,  # Python 3
    }
else:
    compiler_directives = {}

ext_modules = [
    Extension(
        name=module_path,
        sources=[
            str(
                pathlib.Path("src", *module_path.split(".")).with_suffix(
                    ".pyx" if args.use_cython else ".cpp" if language == "c++" else ".c"
                )
            )
        ],
        language=language,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[numpy.get_include()] + extra_include_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for module_path, language, extra_include_dirs in [
        ("dapy.integrators.implicit_midpoint", "c", []),
        ("dapy.integrators.lorenz_1963", "c", []),
        ("dapy.integrators.lorenz_1996", "c", []),
        ("dapy.integrators.interpolate", "c", []),
        ("dapy.ot.costs", "c", []),
        ("dapy.ot.solvers", "c++", [str(pathlib.Path("dapy", "ot"))])
    ]
]

if args.use_cython:
    from Cython.Build import cythonize
    from Cython.Build.Dependencies import default_create_extension

    def create_extension_and_strip_paths_metadata(template, kwds):
        ext, metadata = default_create_extension(template, kwds)
        if "distutils" in metadata:
            if "depends" in metadata["distutils"]:
                del metadata["distutils"]["depends"]
            if "include_dirs" in metadata["distutils"]:
                del metadata["distutils"]["include_dirs"]
        return ext, metadata

    ext_modules = cythonize(
        ext_modules,
        compiler_directives=compiler_directives,
        gdb_debug=args.debug,
        create_extension=create_extension_and_strip_paths_metadata,
    )

setup(ext_modules=ext_modules)
