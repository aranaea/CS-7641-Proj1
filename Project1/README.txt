Code Structure

Most code is broken in to packages named after the algorithm they are exploring.  Each has a main function that can be
run to reproduce the plots and data from the report but they must all be run independently.

All code and data files can be found in https://github.com/aranaea/CS-7641-Proj1 but if you would like to download the
raw data files they can be found at the following locations.

Pendigits: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
Diamonds: https://www.kaggle.com/shivam2503/diamonds

Running the Code
There are very few dependencies and are all captured in the "requirements.txt" file.  You should be able to
run `pip install -r requirements.txt` and then run `PYTHONPATH=. python omscs_7641_p1/<algo>.py` in the source directory.