#!/bin/bash


Input='test_antoniu'
echo " Executing Gauss-Seidel Heat diffusion kernel with input : ${Input}"
./gauss_omp data/${Input}.dat ./Output/${Input}.ppm 

Input='testgrind2'
echo " Executing Gauss-Seidel Heat diffusion kernel with input : ${Input}"
./gauss_omp data/${Input}.dat ./Output/${Input}.ppm 

Input='testgrind2-regions'
echo " Executing Gauss-Seidel Heat diffusion kernel with input : ${Input}"
./gauss_omp data/${Input}.dat ./Output/${Input}.ppm 
