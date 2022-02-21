from setuptools import setup, find_packages

setup(
    name="neuraleduseg",
    version="1.0.0",
    description="Discourse segmentation",
    license="Apache License 2.0",
    url="https://github.com/rknaebel/NeuralEDUSeg",
    packages=find_packages(),
    author="Rene Knaebel",
    author_email="rene.knaebel@uni-potsdam.de",
    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Operating System :: Unix",
                 "Operating System :: MacOS",
                 "Programming Language :: Python :: 3",
                 "Topic :: Text Processing :: Linguistic"],
    keywords="discourse NLP linguistics")
