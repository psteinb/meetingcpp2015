PANDOC ?= pandoc

# Pandoc filters.
FILTERS = $(wildcard tools/filters/*.py)

all : index.html

tools/filters/linkTable : tools/filters/linkTable.hs
	ghc $<

index.html : slides.md links.md tools/filters/linkTable
	${PANDOC} -s --highlight-style=espresso --template=pandoc-revealjs.template -t revealjs -o $@ -V revealjs-width:1600 --section-divs --filter tools/filters/columnfilter.py $< links.md

links.md : slides.md
	${PANDOC} -t json $< | ./tools/filters/dump_links.py > $@

clean :
	rm -f links.md index.html
