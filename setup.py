import setuptools

# Read version
with open('VERSION','r') as fin:
    VERSION = fin.read()

# Setup
setuptools.setup (
    name='tf2torch',
    version=VERSION
)
