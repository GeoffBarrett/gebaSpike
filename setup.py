import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages()
print('found these packages:', pkgs)

pkg_name = "gebaSpike"

setuptools.setup(
    name=pkg_name,
    version="1.0.0",
    author="Geoffrey Barrett",
    author_email="geoffrey.m.barrett@gmail.com",
    description="gebaSpike - GUI designed to improve manual spike sorting of in-vivo electrophysiology data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoffBarrett/gebaSpike.git",
    packages=pkgs,
    install_requires=
    [
        'PyQtGraph',
        'PyQt5',
        'scipy',
        'PyOpenGl',
        'sklearn'
    ],
    package_data={'gebaSpike': ['img/*.png']},
    classifiers=[
        "Programming Language :: Python :: 3.7 ",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
    ],
)
