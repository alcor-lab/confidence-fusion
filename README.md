# Confidence driven TGV fusion


This repository contains the source code corresponding to the method presented in the paper:

* **Confidence driven TGV fusion**, V. Ntouskos, F. Pirri, arXiv: 	arXiv:1603.09302, 2016.

###Contact: 

Valsamis Ntouskos <ntouskos@diag.uniroma1.it>, [http://www.diag.uniroma1.it/~ntouskos](http://www.diag.uniroma1.it/~ntouskos);

ALCOR Lab <alcor@diag.uniroma1.it> [http://www.diag.uniroma1.it/~alcor](http://www.diag.uniroma1.it/~alcor), 


##Instructions##

To run the program you first need to add the 'external' folderin the Matlab path. 

##Quickstart
1. Compile the CUDA kernels located in the `./cuda` folder. Under Windows you can use the provided Makefile: `nmake -f Makefile.win all`;

1. Execute the demo script:
`demo_kitti`.

## External dependencies 

You need the CUDA Toolkit (version 6 or later) and a compatible compiler in order to compile the CUDA kernels. 