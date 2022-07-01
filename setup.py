import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qiskit_helper_functions",
    version="0.0.1",
    description="Quantum Helper Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weiT1993/qiskit_helper_functions.git",
    packages=setuptools.find_packages(),
    author="weiT1993",
    author_email="tangwei1027@gmail.com",
    license="MIT",
    python_requires=">=3.7",
    zip_safe=False,
)
