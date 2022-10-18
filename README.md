# neurobooth-explorer
Neurobooth Explorer - a web app to explore Neurobooth data

##### To create environment from yaml do:
```conda env create -f environment.yml```

##### To create environment manually:
Create environment with desired name - python versions 3.7 to 3.9 tested, v3.10 has unresolved incompatibilities.

Install packages in following order from conda channels, avoid pip:
1. Install ```psycopg2=2.9.3``` from any conda channel, avoid pip. At the time of writing, this version was only available on conda-forge and not on pkgs/main.
2. Install ```dash=1.19.0``` from any conda channel. This installs the following:
  1. ```dash-core-components-1.3.1```
  2. ```dash-html-components-1.0.1```
  3. ```dash-renderer-1.1.2```
  4. ```flask-2.2.2``` (could also be another version of Flask)
3. **App will not work with other dash package versions - only works with 1.19.0**
4. Install ```dash-auth=1.4.1``` from any conda channel.
5. Install from any conda channel:
  1. ```sshtunnel```
  2. ```numpy```
  3. ```pandas``` (tested till v1.5.0)
  4. ```scipy```
6. Install ```h5io``` (v0.1.7 tested)
7. Install neurobooth-terra by ```pip install -e git+https://github.com/neurobooth/neurobooth-terra.git#egg=neurobooth_terra```
8. Check version of ```openssl```, if it is not ```1.1.1q```, do ```conda install -c conda-forge openssl=1.1.1q``` (this resolves clash between openssl and cryptography due to bugs)
9. Check version of ```werkzeug```, if it is not ```2.0.3```, do ```conda install -c conda-forge werkzeug=2.0.3```
  1. This will downgrade Flask to ```flask 2.2.2 --> 2.1.3```

##### Activate conda environment and test
**Run explorer by doing: ```python neurobooth_explorer.py```**
