from distutils.core import setup

from setuptools import setup

def build_native(spec):
    # build an example rust library
    build = spec.add_external_build(
        cmd=['cargo', 'build', '--release'],
        path='./rust'
    )

    spec.add_cffi_module(
        module_path='node2vec._native',
        dylib=lambda: build.find_dylib('node2vec', in_path='target/release'),
        header_filename=lambda: build.find_header('node2vec.h', in_path='target'),
        rtld_flags=['NOW', 'NODELETE']
    )

setup(
    name='node2vec',
    packages=['node2vec'],
    version='0.1.2',
    description='Implementation of the node2vec algorithm.',
    author='Elior Cohen',
    author_email='',
    license='MIT',
    url='https://github.com/eliorc/node2vec',
    setup_requires=['milksnake'],
    install_requires=[
        'networkx',
        'gensim',
        'numpy',
        'tqdm',
        'joblib',
        'milksnake'
    ],
    keywords=['machine learning', 'embeddings'],
    zip_safe=False,
    platforms='any',
    milksnake_tasks=[
        build_native
    ]
)
