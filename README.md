# covid19_xray_analysis

### Bronze medal solution (94th place over 1300 teams) for https://www.kaggle.com/c/siim-covid19-detection/overview
## Installation

First of all, you should have python 3.x to work with this project. The recommended Python version is 3.6 or greater.

Note for Windows users: You should start a command line with administrator's privileges.

First of all, clone the repository:

    git clone https://github.com/greylord1996/covid19_xray_analysis.git
    cd covid19_xray_analysis/

Create a new virtual environment:

    # on Linux:
    python -m venv covenv
    # on Windows:
    python -m venv covenv

Activate the environment:

    # on Linux:
    source covenv/bin/activate
    # on Windows:
    call covenv\Scripts\activate.bat

Install required dependencies:

    # on Linux:
    pip install -r requirements.txt
    # on Windows:
    python -m pip install -r requirements.txt


## Data

To use this code you need to download data and specify paths on it:

- Kaggle input data: https://www.kaggle.com/c/siim-covid19-detection/data
- Resized input images: https://drive.google.com/drive/u/1/folders/1F0oSOKRN5WP-0QMhAvSj4ij7c0kfF8DA
- Public Kernel with trained weights: https://www.kaggle.com/greylord1996/effnetb4-effnetv2-retinanet-nmw

