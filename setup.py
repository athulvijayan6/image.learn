import sys

from pip.download import PipSession
from pip.req import parse_requirements
from setuptools import find_packages, setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 5)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("Unsupported python version.")
    sys.exit(1)


EXCLUDE_FROM_PACKAGES = []
install_requirements = parse_requirements("requirements.txt", session=PipSession())
requirements = [str(ir.req) for ir in install_requirements]
setup(
    name='imagelearn',
    version='1.0.0',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='',
    author='Athul Vijayan',
    author_email='athulvijayan6@gmail.com',
    description='A repository of machine learning algorithms for image learning.',
    license='GPL',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Framework :: tensorflow',
        'Intended Audience :: Researchers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
