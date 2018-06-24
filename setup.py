from setuptools import setup, find_packages

# parametros para Google Cloud ML Engine
setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Config para keras en cloud-ml',
      author='Justo Nombela',
      author_email='justonombela@gmail.com',
      license='MIT',
      install_requires=[
            'keras==2.1.3',
            'h5py',
            'matplotlib',
            'Pillow',
      ],
      zip_safe=False)
    
      

