init:
	pip install -r requirements.txt

testinit:
	pip install virtualenv
	python -m venv --clear .venv
	.\.venv\Scripts\Activate.bat
	python.exe -m pip install --upgrade pip
	pip install pytest

test:
	.\.venv\Scripts\Activate.bat
	python -m pytest
