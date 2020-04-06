#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import setuptools


try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

DISTNAME = "pycombat"
DESCRIPTION = "Python version of data harmonisation technique Combat "
VERSION = "0.13"
AUTHOR = "CoAxLab"
AUTHOR_EMAIL = "jrasero.daparte@gmail.com"
URL = "https://github.com/CoAxLab/pycombat"
DOWNLOAD_URL = URL + "/archive/" + VERSION + ".tar.gz"
    

if __name__ == "__main__":
    setuptools.setup(
         name=DISTNAME,
         version=VERSION,
         packages=[DISTNAME] ,
         author=AUTHOR,
         author_email=AUTHOR_EMAIL,
         description=DESCRIPTION,
         long_description=long_description,
         long_description_content_type='text/markdown',
         url=URL,
         license='MIT',
         install_requires=['numpy','pandas']    
     )
