### ReScience Replication Code

#### Manuscript

The article is in article. To compile the PDF run  

```
cd article
make
```

#### Code repository

This provided code will reproduce all figures presented in the article. Figures are stored in the figures directory

Code has only been tested using python3.6 on linux and macOS

The python dependencies are listed in `requirements.txt` and can be installed with

`pip install -r requirements.txt`

There is also a non-python dependency for swig which must be installed before running the code.

In order to reproduce a particular figure, i.e. figure 1

`python run.py f1`

Figures 1,2,3,4, and 6 can be created.

In order to reproduce all figures

`python run.py all`

or 

`python run.py`

Note: when running figures you may see warnings about "test function going crazy" this is normal and does not affect 
the results.