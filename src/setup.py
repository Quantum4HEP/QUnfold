from setuptools import setup, find_packages


def get_requirements():
    with open("./requirements.txt", "r") as file:
        reqs = file.read().splitlines()
    return reqs


if __name__ == "__main__":

    setup(
        name="QUnfold",
        version="0.3.0",
        author="Gianluca Bianco, Simone Gasperini",
        author_email="biancogianluca9@gmail.com, simone.gasperini4@unibo.it",
        url="https://github.com/JustWhit3/QUnfold/tree/main/src/QUnfold",
        python_requires=">=3.8",
        packages=find_packages(),
        install_requires=get_requirements(),
        license="MIT",
        license_files="../LICENSE",
        long_description_content_type="text/markdown",
        long_description="../README.md",
    )
