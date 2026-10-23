all: main.pdf

main.pdf: main.tex refs.bib appendix_proofs.tex $(wildcard sections/*.tex)
	latexmk -pdf main.tex

clean:
	latexmk -C main.tex

.PHONY: all clean
