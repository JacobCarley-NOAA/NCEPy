from setuptools import setup 

setup(
  name              = "ncepy",
  version           = "0.1",
  description       = "Useful toolkit for analysis and visualization of weather, climate, and ocean data.",
  long_description  = """ This is a toolkit for analysis and visualization of weather, climate, and ocean data.  It is not a package which creates graphics or plots, but rather helps augment such packages.  In this toolkit users will find routines which rotate grid relative winds to earth relative, generate color tables, identify corners for plotting regions, convert units, etc. """,
  url               = "https://github.com/jrcarley/NCEPy",
  author            = "Jacob Carley",
  author_email      = "jacob.carley@noaa.gov",
  platforms         = ["any"],
  license           = "GPL v2",
  classifiers       = ["Development Status :: 3 - Alpha",
                       "Intended Audience :: Science/Research", 
                       "License :: OSI Approved :: GNU General Public License v2 (GPLv2)", 
                       "Programming Language :: Python",
                       "Topic :: Software Development :: Libraries :: Python Modules",
                       "Operating System :: OS Independent"],
  packages          = ['ncepy'],
  )


#Development Status :: 1 - Planning
#Development Status :: 2 - Pre-Alpha
#Development Status :: 3 - Alpha
#Development Status :: 4 - Beta
#Development Status :: 5 - Production/Stable
#Development Status :: 6 - Mature
#Development Status :: 7 - Inactive

