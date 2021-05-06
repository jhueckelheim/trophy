# Makefile for 


HEADER = paper
PICS = images
BIB = references.bib
TEX = .

TARGETS: $(HEADER).pdf

tex_files = $(shell find $(TEX) -name '*.tex' -print)
bib_files = $(shell find $(BIB) -name '*.bib' -print)
pic_files = $(shell find $(PICS) \
		\( -name '*.eps' -print \) -or \( -name '*.epsi' -print \) \
		-or \( -name '*.ps' -print \) -or \( -name '*.png' -print \) \
		-or \( -name '*.fig' -print \) -or \( -name '*.pdf' -print \) \
	)

%.pdf: %.tex $(tex_files) $(bib_files) $(pic_files)
	pdflatex -shell-escape $(HEADER)
	bibtex $(HEADER)
	pdflatex -shell-escape $(HEADER)	
	pdflatex -shell-escape $(HEADER)	
#	pdflatex -shell-escape $(TR)	

clean:
	find . \( -name '*.blg' -print \) -or \( -name '*.aux' -print \) -or \
		\( -name '*.bbl' -print \) -or \( -name '*~' -print \) -or \
		\( -name '*.spl' -print \) -or \
		\( -name '*.bcf' -print \) -or \
		\( -name '*.out' -print \) -or \
		\( -name '*.out.ps' -print \) -or \
		\( -name '*.thm' -print \) -or \
		\( -name '*.lof' -print \) -or \( -name '*.lot' -print \) -or \
		\( -name '*.log' -print \) -or \( -name '*.brf' -print \) -or \
		\( -name '*.toc' -print \) -or \( -name '*.toc.old' -print \) | xargs rm -f; \
	rm -f $(HEADER).dvi $(HEADER).log $(HEADER).ps $(HEADER).pdf $(HEADER).out \
		_region_* TAGS 
