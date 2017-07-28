Separake: Echo-aware source separation
======================================

This repository contains all the code to reproduce the results of the paper
[*Separake: Echo-aware source separation*](http://lcav.epfl.ch).

TBA

We are available for any question or request relating to either the code
or the theory behind it. Just ask!

Abstract
--------

TBA

Authors
-------

* Antoine Deleforge (INRIA)
* Ivan Dokmanić (UIUC)
* Robin Scheibler (EPFL)

<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">

#### Contact

[Robin Scheibler](mailto:robin[dot]scheibler[at]epfl[dot]ch) <br>
EPFL-IC-LCAV <br>
BC Building <br>
Station 14 <br>
1015 Lausanne

Recreate the figures and sound samples
--------------------------------------

In a terminal, run the following script.

    ./make_all_figures.sh

Data used in the paper
----------------------

TBA

Recorded Data
-------------

The recorded samples are stored in the `recordings` folder.
Detailed description and instructions are provided along the data.

Overview of results
-------------------

TBA

Dependencies
------------

The script `system_install.sh` was used to install all the required software on a blank UBUNTU Xenial server.

* A working distribution of [Python 3.5](https://www.python.org/downloads/) (but 2.7 should work too).
* [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/)
* We use the distribution [anaconda](https://store.continuum.io/cshop/anaconda/) to simplify the setup of the environment.
* Computations are very heavy and we use the
  [MKL](https://store.continuum.io/cshop/mkl-optimizations/) extension of
  Anaconda to speed things up. There is a [free license](https://store.continuum.io/cshop/academicanaconda) for academics.
* We used ipyparallel and joblib for parallel computations.
* [matplotlib](http://matplotlib.org) and [seaborn](https://stanford.edu/~mwaskom/software/seaborn/index.html#) for plotting the results.

The pyroomacoustics is used for STFT, fractionnal delay filters, microphone arrays generation, and some more.

    pip install pyroomacoustics

List of standard packages needed

    numpy, scipy, pandas, ipyparallel, seaborn, zmq, joblib, samplerate


Systems Tested
--------------

###Linux

| Machine | ICCLUSTER EPFL                  |
|---------|---------------------------------|
| System  | Ubuntu 16.04.5                  |
| CPU     | Intel Xeon E5-2680 v3 (Haswell) |
| RAM     | 64 GB                           |

###OS X

| Machine | MacBook Pro Retina 15-inch, Early 2013 |
|---------|----------------------------------------|
| System  | OS X Maverick 10.11.6                  |
| CPU     | Intel Core i7                          |
| RAM     | 16 GB                                  |

    System Info:
    ------------
    Darwin 15.6.0 Darwin Kernel Version 15.6.0: Mon Aug 29 20:21:34 PDT 2016; root:xnu-3248.60.11~1/RELEASE_X86_64 x86_64

    Python Info:
    ------------
    Python 2.7.11 :: Anaconda custom (x86_64)

    Python Packages Info (conda)
    ----------------------------
    # packages in environment at /Users/scheibler/anaconda:
    accelerate                2.0.2              np110py27_p0  
    accelerate_cudalib        2.0                           0  
    anaconda                  custom                   py27_0  
    ipyparallel               5.0.1                    py27_0  
    ipython                   4.2.0                    py27_0  
    ipython-notebook          4.0.4                    py27_0  
    ipython-qtconsole         4.0.1                    py27_0  
    ipython_genutils          0.1.0                    py27_0  
    joblib                    0.9.4                    py27_0  
    mkl                       11.3.3                        0  
    mkl-rt                    11.1                         p0  
    mkl-service               1.1.2                    py27_2  
    mklfft                    2.1                np110py27_p0  
    numpy                     1.11.0                    <pip>
    numpy                     1.11.1                   py27_0  
    numpydoc                  0.5                       <pip>
    pandas                    0.18.1              np111py27_0  
    pyzmq                     15.2.0                   py27_1  
    scikits.audiolab          0.11.0                    <pip>
    scikits.samplerate        0.3.3                     <pip>
    scipy                     0.17.0                    <pip>
    scipy                     0.18.0              np111py27_0  
    seaborn                   0.7.1                    py27_0  
    seaborn                   0.7.1                     <pip>

License
-------

Copyright (c) 2016, Antoine Deleforge, Ivan Dokmanić, Robin Scheibler

All the code in this repository is under MIT License.

