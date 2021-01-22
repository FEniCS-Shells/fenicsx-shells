from skbuild import setup

setup(
    name="fenics-shellsx",
    version="0.0.1",
    description="FEniCS-ShellsX,
    author="Jack S. Hale",
    license="LGPLv3",
    packages=["fenics_shellsx"],
    package_dir={"": "cpp"},
    cmake_install_dir="cpp/fenics_shellsx",
)
