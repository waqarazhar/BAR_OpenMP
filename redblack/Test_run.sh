#!/bin/bash

Input='test_redblack'
echo " Executing redblack Heat diffusion kernel with input : ${Input}"
./redblack_omp data/${Input}.dat ./output/${Input}.ppm 

Input='test_redblack2'
echo " Executing redblack Heat diffusion kernel with input : ${Input}"
./redblack_omp data/${Input}.dat ./output/${Input}.ppm 

