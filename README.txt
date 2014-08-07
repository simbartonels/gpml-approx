The code I created is placed in project/ directory.
gpml contains the gpml code; it's Matlab so everyone with Matlab should be able to use it. 
figtree-0.9.3/ contains the IFGT transform code. It's written in C++, so you'll probably need to compile it on your machine separately. In the
project we use Matlab wrappers provided with this code, so you need to compile to mex files in figtree-0.9.3/matlab/ dir. The relevant documentation
is placed in the .m files in that dir.

NOTE: I modified some files in the gpml package: in particular
gpml/gp.m           has added code to measure runtimes of interest
gpml/inf/{infExact.m,infFITC.m} are modified to use less space and some other small modifications/fixes were added. See README in the gpml directory.

Krzysztof Chalupka, 2011-10-07

