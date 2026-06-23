OUTDIR = out
MAIN = Kazachkov_BSc_diploma_presentation

.PHONY: all clean

all: $(OUTDIR)/$(MAIN).pdf

$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex | $(OUTDIR)
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(MAIN).tex

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	LC_ALL=C LANG=C latexmk -C -outdir=$(OUTDIR) $(MAIN).tex
