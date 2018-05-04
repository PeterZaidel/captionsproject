#!/bash/bin
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
sudo make install
sudo python setup.py install