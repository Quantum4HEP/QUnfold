from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="QUnfold",
        packages=find_packages(include=["QUnfold"]),
        version="0.0.2",
        author="Gianluca Bianco, Simone Gasperini",
        author_email="biancogianluca9@gmail.com, simone.gasperini4@unibo.it",
        url="https://github.com/JustWhit3/QUnfold/tree/main/src/QUnfold",
        python_requires=">=3.8",
        install_requires=[
            "dwave-ocean-sdk",
            "pyqubo",
            "matplotlib",
            "scipy",
        ],
    )
