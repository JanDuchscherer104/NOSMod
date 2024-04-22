from setuptools import find_packages, setup

setup(
    name="dl_modulator",
    version="0.1.0",
    description="A DL based modulator for optical communication to allow ultra narrow filtering",
    author="Jan Duchscherer",
    author_email="jan.duchscherer@hm.edu",
    url="https://github.com/yourusername/dl-modulator",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
)
