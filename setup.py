from setuptools import find_packages, setup

# ----------------------------- Setup -----------------------------
setup(
    name="flag_gems",
    version="2.2",
    authors=[
        {"name": "Zhixin Li", "email": "strongspoon@outlook.com"},
        {"name": "Tongxin Bai", "email": "waffle.bai@gmail.com"},
        {"name": "Yuming Huang", "email": "jokmingwong@gmail.com"},
        {"name": "Feiyu Chen", "email": "iclementine@outlook.com"},
    ],
    description="FlagGems is a function library written in Triton.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8.0",
    license="Apache Software License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[],
    extras_require={
        "test": [
            "pytest>=7.1.0",
            "numpy>=1.26",
            "scipy>=1.14",
        ],
        "example": [
            "transformers>=4.40.2",
        ],
    },
    url="https://github.com/FlagOpen/FlagGems",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,  # To include non-Python files, e.g., README
    package_data={
        "flag_gems.runtime": ["**/*.yaml"],
    },
)
