

SRC_FILES=$(wildcard *tex)
DST_FILES=$(SRC_FILES:.tex=.svg)

all : $(DST_FILES)

%.pdf : %.tex
	pdflatex $<

%.svg : %.pdf
	pdf2svg $< $@
