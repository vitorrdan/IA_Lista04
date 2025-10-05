from setuptools import setup, find_packages

setup(
    name='decision_tree_lib',
    version='0.1.0',
    author='Daniel Vítor',
    description='Implementação do zero dos algoritmos ID3, C4.5 e CART.',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)