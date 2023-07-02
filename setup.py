from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='bnFloodDetector',
      version='0.1.0',
      description='Flood Detector',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Intended Audience :: Developers',
      ],
      keywords='flood detector segmentation effientnet',
      url='https://github.com/ahestevenz/flood-detection-segmentation',
      author='Ariel Hernandez <ahestevenz@bleiben.ar>',
      author_email='ahestevenz@bleiben.ar',
      license='Proprietary',
      install_requires=[
          'numpy==1.23.0', 'pathlib', 'pandas==1.4.3', 'scikit-learn',
          'matplotlib==3.5.2', 'loguru==0.6.0', "opencv-python==4.6.0.66",
          "gdown==4.5.1", "lxml", "segmentation-models-pytorch", "albumentations",
          "seaborn"
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts': ['bn-run-train=bnFloodDetector.scripts.run_train:main',
                              'bn-run-test=bnFloodDetector.scripts.run_test:main',
                              ],
      },
      include_package_data=True,
      zip_safe=True
      )
