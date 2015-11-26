PANDOC ?= pandoc

# Pandoc filters.
FILTERS = $(wildcard tools/filters/*.py)

all : slides.html

tools/filters/linkTable : tools/filters/linkTable.hs
	ghc $<

slides.html : slides.md tools/filters/linkTable
	${PANDOC} $< -s --highlight-style=espresso --template=pandoc-revealjs.template -t revealjs -o $@ -V revealjs-width:1600 --section-divs --filter tools/filters/columnfilter.py --filter tools/filters/linkTable
