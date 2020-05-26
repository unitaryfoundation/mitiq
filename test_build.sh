#!/usr/bin/env bash

if [ "$1" != "-docs" ]; then
    echo "Running Tests...";
    pytest --cov=mitiq mitiq/tests mitiq/benchmarks/tests mitiq/mitiq_qiskit/tests;
elif [ "$1" != "-tests" ]; then
    echo "Building and Testing Docs...";
    cd docs && make html;
    make doctest;
fi
