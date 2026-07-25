# Asimov = median of toys -- HWW meeting, 22/07/2026
#
#   make        build slides.pdf
#   make clean  remove the LaTeX by-products (keeps the PDF)
#   make purge  remove the by-products and the PDF
#
# lualatex is required: the theme uses fontspec and unicode-math.
# The CERN texlive tree is prepended to PATH here, so this works from a bare
# shell with no lsetup and no container.

DECK    := slides
TEXLIVE := /cvmfs/sft.cern.ch/lcg/external/texlive/2025/bin/x86_64-linux
THEME   := ../beamer/beamerthememarta.sty

export PATH := $(TEXLIVE):$(PATH)
export TEXINPUTS := ../beamer:

.PHONY: all clean purge

all: $(DECK).pdf

$(DECK).pdf: $(DECK).tex $(THEME)
	latexmk -lualatex -interaction=nonstopmode -halt-on-error $(DECK)

clean:
	latexmk -c $(DECK)

purge:
	latexmk -C $(DECK)
