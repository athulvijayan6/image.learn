import os
import sys
from distutils.sysconfig import get_python_lib

from setuptools import find_packages, setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 5)

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of image.learn requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)


EXCLUDE_FROM_PACKAGES = []


# Dynamically calculate the version based on django.VERSION.
version = __import__('django').get_version()


setup(
    name='image.learn',
    version=version,
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='',
    author='Django Software Foundation',
    author_email='foundation@djangoproject.com',
    description=('A high-level Python Web framework that encourages '
                 'rapid development and clean, pragmatic design.'),
    license='BSD',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    scripts=['django/bin/django-admin.py'],
    entry_points={'console_scripts': [
        'django-admin = django.core.management:execute_from_command_line',
    ]},
    install_requires=['pytz'],
    extras_require={
        "bcrypt": ["bcrypt"],
        "argon2": ["argon2-cffi >= 16.1.0"],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Internet :: WWW/HTTP :: WSGI',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)


if overlay_warning:
    sys.stderr.write("""
========
WARNING!
========
You have just installed Django over top of an existing
installation, without removing it first. Because of this,
your install may now include extraneous files from a
previous version that have since been removed from
Django. This is known to cause a variety of problems. You
should manually remove the
%(existing_path)s
directory and re-install Django.
