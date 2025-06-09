from setuptools import find_packages, setup

def get_requirements(file_path):
    """
    This function returns a list of requirements from the given file path,
    excluding '-e .' and empty lines.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

    # Strip and clean up each requirement
    requirements = [req.strip() for req in requirements if req.strip()]

    # Remove editable mode if present
    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Abdul Momin Khan",
    author_email="khanabdulmomin24@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
