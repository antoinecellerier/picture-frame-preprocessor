from setuptools import setup, find_packages

setup(
    name="frame-prep",
    version="0.2.1",
    description="Intelligent image preprocessor for e-ink picture frames with contextual zoom and OpenVINO acceleration",
    author="Antoine",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "frame-prep=frame_prep.cli:cli",
        ],
    },
)
