#!/bin/bash

Rscript -e "rmarkdown::render('ProjectWork_Palleschi.Rmd', output_format='pdf_document')"
Rscript -e "rmarkdown::render('ProjectWork_Palleschi.Rmd', output_format='md_document', output_file='README.md')"
