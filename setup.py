from setuptools import setup, find_packages

setup(
    name="inferx",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "httpx>=0.24.0",
        "psutil>=5.9.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "nvidia-ml-py>=12.0.0",
        "huggingface-hub>=0.14.0",
    ],
    entry_points={
        "console_scripts": [
            "inferx=inferx.main:main",
        ],
    },
)
