# README #

This repository contains Python3 package for picking up the real candidates from GOTO SCIENCE image. It contains two real-bogus classifiers built by using Random Forest (RF) classifier and Artificial Neural Network (ANN).

### Dependencies ###

* [astroasciidata](https://github.com/ewittmyl/astroasciidata)
* [alipy](https://github.com/ewittmyl/alipy)
* [HOTPANTS](https://github.com/ewittmyl/hotpants)
* SExtractor
* [RF Classifier](https://my.pcloud.com/publink/show?code=XZLWrA7ZhwqB3gRkIkH4o3VQXXoPIbnAwmdV)
* [GOTOcat](https://github.com/GOTO-OBS/gotocat)
* [PyMPChecker](https://github.com/GOTO-OBS/PyMPChecker)

### How to install? ###

* Clone the `g-TranRec` repository
* Run `pip install -r requirements.txt` 
* Install all the dependencies listed above
* Put the RF Classifier into `g-TranRec/gTranRec/data`
* Unzip all the zip files in `g-TranRec/gTranRec/data`
* Export the environment variable `$HP_PATH` for HOTPANTS
* Install `g-TranRec` using pip:
```
pip install .
```

See the tutorial notebook (in the doc subdirectory) for more usage suggestions

### Contact ###

* yik.mong@monash.edu
