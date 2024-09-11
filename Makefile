#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = enfify
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 enfify
	isort --check --diff --profile black enfify
	black --check --config pyproject.toml enfify

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml enfify




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"



## Try enfify_alpha
.PHONY: try
try:
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) enfify/enfify_alpha.py data/samples/whu_cut_min_001_ref.wav




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: dataset
dataset: # requirements
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) enfify/data/make_dataset.py


## Make Augmentation
.PHONY: augmentation
augmentation: # requirements # MayDo dataset as requirement?
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) enfify/data/make_augmentation.py


# ## Make Preprocessing
# .PHONY: preprocessing
# preprocessing: # requirements # MayDo augmentation as requirement?
# 	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) # enfify/data/make_augmentation.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python3 -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
