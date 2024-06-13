import site
import sys

# Retrieve the site-packages path
site_packages_path = site.getsitepackages()[0] if hasattr(site, 'getsitepackages') else site.getusersitepackages()

# Retrieve the path to the Python executable
python_executable_path = sys.executable

print(f"-DPYTHON_LIBRARY_DIR=\"{site_packages_path}\" -DPYTHON_EXECUTABLE=\"{python_executable_path}\"")

