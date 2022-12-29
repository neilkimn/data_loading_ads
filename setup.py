from setuptools import setup

setup(
    name='ads',
    install_requires=[
        'pip',
        'pytorch',
        'torchvision',
        'pytorch-cuda==11.6',
        'black',
        'tensorboardX',
        'opencv-python',
        'torch-tb-profiler',
        'nvidia-pyindex',
        'nvidia-dali-cuda110',
        "matplotlib",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cu116",
        "https://developer.download.nvidia.com/compute/redist"
    ]
)
