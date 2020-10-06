#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./dev/linter.sh" before you commit

vergte() {
  [ "$2" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

{
	black --version | grep "19.3b0" > /dev/null
} || {
	echo "Linter requires black==19.3b0 !"
	exit 1
}

ISORT_TARGET_VERSION="4.3.21"
ISORT_VERSION=$(isort -v | grep VERSION | awk '{print $2}')
vergte "$ISORT_VERSION" "$ISORT_TARGET_VERSION" || {
  echo "Linter requires isort>=${ISORT_TARGET_VERSION} !"
  exit 1
}

set -v

echo "Running isort ..."
isort -y -sp . --atomic

echo "Running black ..."
black .

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && arc lint
