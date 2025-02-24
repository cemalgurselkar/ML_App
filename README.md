# ML Training APP

The purpose of this project is that aim to learn how use flask and streamlite and how build a simple web app with this.
In this project, The user can choice one of three models (SVR, XBGRegressin, MLPRegresion) and can train their data.
Also, the user can do data scaling, data spearation, cross validation.
If the user decide their model is good for using, they can install as a .pickle format and can test their own model.

**Note: You can upload the .csv or .xlsx extension file you want. But, the data must be for regression problem because the models in project are regression models.**

**Note: For this project, python3.10.2 version was used.**

## How to clone and run this repo?
### To Clone
```bash
git clone ML_App
cd ML_App
```
### To prepare before run the project.
```python
python3 -m venv venv # FOR python3.10.2 version
source venv/bin/activate
pip install -r requirement.txt # to install the required libraries.
```

### How to run?
#### Why main.py?
**This script starts a Flask API (model_api.py), runs the Streamlit app (app2.py), and terminates the Flask process once the Streamlit app closes.**

```python
python3 main.py
```