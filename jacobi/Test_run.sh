#!/bin/bash


Input='test_jacobi'
echo " Executing Jacobi Heat diffusion kernel with input : ${Input}"
./jacobi_omp data/${Input}.dat ./Output/${Input}.ppm 

Input='testgrind'
echo " Executing Jacobi Heat diffusion kernel with input : ${Input}"
./jacobi_omp data/${Input}.dat ./Output/${Input}.ppm 

