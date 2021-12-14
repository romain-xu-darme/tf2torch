import setuptools

# Read version
with open('VERSION','r') as fin:
    VERSION = fin.read()

# Read README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


# Setup
setuptools.setup (
    name='tf2torch',
    version=VERSION,
    description='Tensorflow to Pytorch model converter.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    entry_points={
        'console_scripts': [
            'tf2torch_convert  = convert:_main',
        ],
    },
)
