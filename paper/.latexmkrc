# Build: latexmk  (run from the paper/ folder; VS Code's LaTeX Workshop does this too).
# All outputs (PDF, .aux, .log, .bbl, ...) go to paper/build/ so the source folder stays clean.
use Cwd 'getcwd';

$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'build';
@default_files = ('main.tex');

# BibTeX runs inside build/, so tell it where tmlr.bst and references.bib live (= this folder).
my $paper_dir = getcwd();
ensure_path('BSTINPUTS', $paper_dir);   # TeX Live
ensure_path('BIBINPUTS', $paper_dir);
$bibtex = "bibtex -include-directory=\"$paper_dir\" %O %S" if $^O =~ /MSWin32|msys|cygwin/;   # MiKTeX (Windows)
