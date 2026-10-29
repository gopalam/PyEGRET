# Load the R sources of EGRET 3.0.12 and EGRETci 2.0.5 kept in reference/,
# without installing either package. Run from this folder.
#
# Needs R with the survival and truncnorm packages, and binom for wBT
# (install.packages(c("survival", "truncnorm", "binom"))).
suppressMessages({
  library(survival)
  library(truncnorm)
})
ref <- normalizePath(file.path("..", "..", "..", "..", "reference"))

prepare <- function(file) {
  txt <- readLines(file, warn = FALSE)
  # The sources call each other through the package namespace; here they
  # live in the global environment instead.
  txt <- gsub("EGRET::", "", txt, fixed = TRUE)
  txt <- gsub("utils::packageVersion(\"EGRET\")", "package_version(\"3.0.12\")", txt,
              fixed = TRUE)
  txt
}
for (f in list.files(file.path(ref, "EGRET_R"), pattern = "\\.[Rr]$", full.names = TRUE)) {
  if (basename(f) == "EGRET.R") next
  eval(parse(text = prepare(f)), envir = globalenv())
}
for (f in list.files(file.path(ref, "EGRETci_R"), pattern = "\\.[Rr]$", full.names = TRUE)) {
  eval(parse(text = prepare(f)), envir = globalenv())
}
load(file.path(ref, "..", "tests", "reference", "Choptank_eList.RData"))
