"""
Summit.OS Python SDK

Official Python SDK for integrating with Summit.OS distributed intelligence fabric.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="summit-os-sdk",
    version="1.0.0",
    author="Big Mountain Technologies",
    author_email="sdk@bigmt.ai",
    description="Python SDK for Summit.OS distributed intelligence fabric",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bigmt/summit-os-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "ros2": [
            "rclpy>=3.0.0",
            "sensor_msgs>=4.0.0",
            "geometry_msgs>=4.0.0",
        ],
        "mqtt": [
            "paho-mqtt>=1.6.0",
        ],
        "websocket": [
            "websocket-client>=1.6.0",
        ],
        "ai": [
            "torch>=1.12.0",
            "onnxruntime>=1.12.0",
            "numpy>=1.21.0",
            "opencv-python>=4.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "summit-os-cli=summit_os.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "summit_os": [
            "models/*.onnx",
            "config/*.yaml",
            "templates/*.json",
        ],
    },
)
