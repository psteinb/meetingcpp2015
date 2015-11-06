PANDOC ?= pandoc

# Pandoc filters.
FILTERS = $(wildcard tools/filters/*.py)

all : slides.html

slides.html : slides.md
	${PANDOC} $< --template=pandoc-revealjs.template -t revealjs -o $@ -V revealjs-width:1600 --section-divs --filter tools/filters/columnfilter.py
