

SRC_FILES=$(wildcard *tex)
DST_FILES=$(SRC_FILES:.tex=.png)

all : $(DST_FILES)

%.pdf : %.tex
	pdflatex $<

%.png : %.pdf
	convert $< $@
