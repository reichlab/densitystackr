## Miscellaneous utility functions
##
## logspace_sub
## logspace_add
## logspace_sum
## logspace_sum_matrix_rows
## logspace_sub_matrix_rows

## interface to R's C API for logspace arithmetic

logspace_sub <- function(logx, logy) {
  return(.Call("logspace_sub_C",
    as.numeric(logx),
    as.numeric(logy),
    PACKAGE = "densitystackr"))
}

logspace_add <- function(logx, logy) {
  return(.Call("logspace_add_C",
    as.numeric(logx),
    as.numeric(logy),
    PACKAGE = "densitystackr"))
}

logspace_sum <- function(logx) {
  dim(logx) <- c(1, length(logx))
  return(logspace_sum_matrix_rows(logx))
}

logspace_sum_matrix_rows <- function(logX) {
  return(.Call("logspace_sum_matrix_rows_C",
    as.numeric(logX),
    as.integer(nrow(logX)),
    as.integer(ncol(logX)),
    PACKAGE = "densitystackr"))
}

logspace_sub_matrix_rows <- function(logX) {
  if(!is.matrix(logX) || !identical(ncol(logX), 2L))
    stop("logX must be a matrix with 2 columns")

  return(.Call("logspace_sub_matrix_rows_C",
    as.numeric(logX),
    as.integer(nrow(logX)),
    PACKAGE = "densitystackr"))
}
