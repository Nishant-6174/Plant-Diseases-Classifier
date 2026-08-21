from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Returns list of requirements parsed from requirements.txt
    """
    requirements = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        lines = file_obj.readlines()
        for line in lines:
            req = line.strip()
            # Ignore comments and empty lines
            if req and not req.startswith("#"):
                if req != HYPHEN_E_DOT:
                    requirements.append(req)
    return requirements


setup(
    name="plant-disease-classifier",
    version="1.0.0",
    author="AgriVision AI Team",
    author_email="contact@agrivision.ai",
    description="End-to-End Deep Learning Plant Disease Classification & Treatment Suite",
    long_description=open("README.md", "r", encoding="utf-8").read() if open("README.md").readable() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
