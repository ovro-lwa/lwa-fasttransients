from setuptools import setup, find_packages

from version import get_git_version

setup(
    name="lwa-fasttransients",
    version=get_git_version(),
    url="https://github.com/ovro-lwa/lwa-fasttransients",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    zip_safe=False,
    # Use scripts= not console_scripts: setuptools entry-point wrappers call
    # load_entry_point() which can raise StopIteration on editable installs in
    # older conda/fasttransients envs even when importlib.metadata sees the EP.
    scripts=["scripts/lwa-voltage-beam"],
)
