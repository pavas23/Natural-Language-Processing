## Steps to Run

### Create virtual env

- python3 -m venv myenv
- source myenv/bin/activate
- cd OpenIE
- python3 -m pip install -r requirements.txt
- python3 -m spacy download en_core_web_sm

## Steps to Run Jupyter Notebook

- Set the ipykernel to use local venv by running this command
```
python3 -m ipykernel install --user --name=myenv
```
