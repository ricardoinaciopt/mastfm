# Package Testing

Instructions to create an isolated environment to test the package presented in the demo at ECMLPKDD 2025.

## Install

First, lets setup the environment. Run the following commands in the provided folder:

```console
$ python -m venv mastfm-test
```

Use only the following set of commands that is relevant to your OS:

```console
# UNIX / MacOS

$ ./mastfm-test/bin/pip install numpy pandas matplotlib ipykernel ipywidgets datasetsforecast xgboost mastfm
$ ./mastfm-test/bin/python -m ipykernel install --user --name=mastfm-test --display-name "Python (mastfm-test)" 

# Windows

$ mastfm-test\Scripts\pip install numpy pandas matplotlib ipykernel ipywidgets datasetsforecast xgboost mastfm
$ mastfm-test\Scripts\python -m ipykernel install --user --name=mastfm-test --display-name "Python (mastfm-test)"
```

Now, activate the environment:

```console
$ source mastfm-test/bin/activate
```

If everything is correct, `(mastfm-test)` should appear in the CLI.
The notebook was tested in **VSCode**, but should work the same in any IDE / code editor.

In the case of the first, simply open it in the current directory using:

```console
$ code .
```

Or open the `.ipynb` file manually.

Make sure you select the correct environment / kernel at the top!

___

## Uninstall

In the end, to clean up everything, run, in the provided folder:

Deactivate the environment:

```console
$ deactivate
```

Use only the following set of commands that is relevant to your OS:

```console
# UNIX / MacOS (add your user name in the path)

$ rm -rf mastfm-test datasets
$ rm -rf /Users/<your-user>/Library/Jupyter/kernels/mastfm-test

# Windows

$ rmdir /s /q mastfm-test
$ rmdir /s /q datasets
$ rmdir /s /q "%APPDATA%\jupyter\kernels\mastfm-test"
```