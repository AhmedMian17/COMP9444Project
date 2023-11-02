<h1>HOW TO SETUP PYTHON ENVIRONMENT (PACKAGES):</h1>

```
python3 -m venv ./venv
```
may be "python" or "py" instead of "python3"
```
./venv/scripts/activate
```
activate your venv on windows, may be different on mac.
```
pip install -r requirements.txt
```
install some relevant scientific packages (should take a few minutes)
```
pip list
```
show what packages were installed to confirm success.


<h1>HOW TO GET BACK INTO YOUR PYTHON ENVIRONMENT:</h1>

```
./venv/scripts/activate
```
activate your venv on windows, may be different on mac.


<h1>IF THE REQUIREMENTS HAVE BEEN CHANGED:</h1>

```
./venv/scripts/activate
```
activate your venv
```
pip install -r requirements.txt
```
will install packages that don't match the requirements.

