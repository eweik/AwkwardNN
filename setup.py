import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="awkwardNN", # Replace with your own username
    version="0.0.1",
    author="Edison Weik",
    author_email="ew1692@nyu.edu",
    description="A Pytorch RNN for awkward-array input",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eweik/AwkwardNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
    python_requires='>=3.6',
)