from pathlib import Path
from setuptools import find_packages, setup

req_path = Path(__file__).parent / "requirements.txt"
requirements = []
if req_path.exists():
    with open(req_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith(("#", "-e"))
        ]

# Setup package definition
setup(
    name="pl-transfermarkt-value-prediction",
    version="0.1.0",
    description="PL transfer market value prediction pipeline",
    author="Aditya Jain",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)