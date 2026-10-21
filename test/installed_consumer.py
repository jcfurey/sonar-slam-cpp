"""Compile and run against installed exported targets, without source includes."""
import os
from pathlib import Path
import subprocess
import sys

prefix, build = sys.argv[1:]
env = dict(os.environ)
env['CMAKE_PREFIX_PATH'] = prefix + os.pathsep + env.get('CMAKE_PREFIX_PATH', '')
source = Path(__file__).parent / 'consumer'
subprocess.run(['cmake', '-S', str(source), '-B', build], env=env, check=True)
subprocess.run(['cmake', '--build', build, '--parallel', '2'], env=env, check=True)
subprocess.run([str(Path(build) / 'installed_sonar_consumer')], env=env, check=True)
