import site
from pathlib import Path
import subprocess

site_packages = Path(site.getsitepackages()[0])
patch_file = site_packages / 'tensorflow_estimator' / 'python' / 'estimator' / 'keras.py'
subprocess.run(['patch', '-b', str(patch_file), 'estimator.patch'])
