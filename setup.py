import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featexp",
    version="0.0.3",
    author="Abhay Pawar",
    author_email="abhayspawar@gmail.com",
    description="Feature exploration for supervised learning",
    long_description="Featexp helps with feature understanding, feature debugging, leakage detection, finding noisy features and model monitoring",
    long_description_content_type="text/markdown",
    url="https://github.com/abhayspawar/featexp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas==0.23.4', 'numpy==1.14.3', 'matplotlib==2.2.2']

)

