import setuptools 

setuptools.setup(
    name='retinapy',
    version='0.0.1.dev6',
    author='Baden Lab members',
    author_email='kevin@kdoran.com',
    description=('RetinaPy is a Python package for working with recordings of '
        'retina activity'),
    packages=['retinapy'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pickle5',
        'h5py'],
    package_dir={'': 'src'},
    include_package_data=True,
    license = 'BSD-3',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research']
)

