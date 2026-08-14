from setuptools import find_packages, setup

package_name = 'vdi_brain'

setup(
    name=package_name,
    version='0.0.0',
    # 🛠️ CHANGED HERE: Hardcoded package name inclusion to bypass discovery errors
    packages=[package_name], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sabari',
    maintainer_email='p.sabarinath.j@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'intention_perception_node = vdi_brain.intention_node:main',
        ],
    },
)