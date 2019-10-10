from setuptools import setup

PACKAGENAME = 'pcafactory'
DESCRIPTION = ''
LONG_DESCRIPTION = ''
AUTHOR = 'Andres Izquierdo'
AUTHOR_EMAIL = 'andres.izquierdo.c@gmail.com'
LICENSE = 'unknown'
URL = 'https://github.com/andizq/pcafactory'
VERSION = '0.0.dev0'
package_info = {}

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=['pcafactory']
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      use_2to3=False,
      python_requires='>={}'.format("2.7"),
      **package_info
)

