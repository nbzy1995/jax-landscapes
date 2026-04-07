from setuptools import setup, find_packages

setup(
    name='jax_landscape',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'jax-md @ git+https://github.com/nbzy1995/jax-md.git',
    ],
    entry_points={
        'console_scripts': [
            'jax_landscape=jax_landscape.main:main',
        ],
    },
)
