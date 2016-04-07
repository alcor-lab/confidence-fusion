# Confidence driven TGV fusion


This repository contains the source code corresponding to the method presented in the paper:

* **Confidence driven TGV fusion**, V. Ntouskos, F. Pirri, arXiv: 	arXiv:1603.09302, 2016.

###Contact: 

Valsamis Ntouskos <ntouskos@diag.uniroma1.it>, [http://www.diag.uniroma1.it/~ntouskos](http://www.diag.uniroma1.it/~ntouskos);

ALCOR Lab <alcor@diag.uniroma1.it> [http://www.diag.uniroma1.it/~alcor](http://www.diag.uniroma1.it/~alcor), 


##Instructions##

To run the demo you need to download the KITTI 2012 multiview dataset (data, calibration files, and multi-view extension) from [here](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo).  

The disparity maps for all the stereo-pairs of the dataset, computed based on the method presented in the paper "Confidence driven TGV fusion", J. Zbontar and Y. LeCun, CVPR, 2015, are available for download from [here](http://www.dis.uniroma1.it/~alcor/site/datasets/kitti_2012_mv_cm-cnn.zip). 

##Quickstart
1. Compile the CUDA kernels located in the `./cuda` folder. Under Windows you can use the provided Makefile: `nmake -f Makefile.win all`;

1. Modify the path variables `basedir` and `w_path` in the demo script `demo_kitti`.

1. Execute the demo script:
`demo_kitti`.

## External dependencies 

You need the CUDA Toolkit (version 6 or later) and a compatible compiler in order to compile the CUDA kernels. 