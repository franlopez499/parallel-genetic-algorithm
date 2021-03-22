#!/bin/bash

make sec


    echo -n "7000 & 256 & 8 &" >> log
    make test_sec_small_256 >> log
    echo -n "20000  & 256 & 8 &" >> log
    make test_sec_big_256 >> log
    echo -n "7000 & 512 & 8 &" >> log
    make test_sec_small_512 >> log
    echo -n "20000 & 512 & 8 &" >> log
    make test_sec_big_512 >> log

    
    echo -n "7000 & 256 & 12 &" >> log
    make test_sec_small_2566 >> log
    echo -n "20000  & 256 & 12 &" >> log
    make test_sec_big_2566 >> log
    echo -n "7000 & 512 & 12 &" >> log
    make test_sec_small_5126 >> log
    echo -n "20000 & 512 & 12 &" >> log
    make test_sec_big_5126 >> log      