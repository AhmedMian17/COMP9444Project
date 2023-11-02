HOW TO SETUP PYTHON ENVIRONMENT (PACKAGES):

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


HOW TO GET BACK INTO YOUR PYTHON ENVIRONMENT:
```
./venv/scripts/activate
```
activate your venv on windows, may be different on mac.


IF THE REQUIREMENTS HAVE BEEN CHANGED:
```
./venv/scripts/activate
```
activate your venv
```
pip install -r requirements.txt
```
will install packages that don't match the requirements.

