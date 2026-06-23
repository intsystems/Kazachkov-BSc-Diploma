OUTDIR = out
OBJDIR = $(OUTDIR)/latexrun

.PHONY: all clean

all: $(OUTDIR)/main.pdf

INCLUDES = include/*.tex

PARTS = parts/*.tex

MAIN = main.tex

$(OUTDIR)/main.pdf: $(MAIN) $(PARTS) $(INCLUDES) references.bib gost71u.bst | $(OUTDIR) $(OBJDIR)
	latexrun/latexrun -O $(OBJDIR) -o $(OUTDIR)/main.pdf main

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR)
