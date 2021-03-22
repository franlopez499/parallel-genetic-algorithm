#!/bin/bash

make sec
echo "Pruebas secuencial" >> log

    echo "" 
    echo "Generaciones: 7000" >> log
    echo "Poblacion: 256" >> log
    make test_sec_small_256 >> log
   