from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chain-of-thought-tool",
    version="0.1.0",
    author="Code Developer", 
    author_email="code-developer@democratize.technology",
    description="A lightweight Chain of Thought reasoning tool for LLM function calling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    keywords="llm, function-calling, reasoning, ai, tools, chain-of-thought, cot",
    project_urls={
        "Source": "https://github.com/democratize-technology/chain-of-thought-tool",
        "Bug Reports": "https://github.com/democratize-technology/chain-of-thought-tool/issues",
    },
)
