library(tidyr)
library(dplyr)
library(xgboost)

context("objective function gradient and (diagonal elements of) hessian")

## make up some data for purposes of the unit tests
## 3 models: "a", "b", "c"
component_models <- letters[1:3]

set.seed(9873)
loso_pred_res <- data.frame(
  model = paste0("log_score_", rep(letters[1:3], each = 100)),
  d = rep(1:100, times = 3),
  loso_log_score = c(
    log(runif(100, 0, 1)), # model a's performance not related to d
    sort(log(runif(100, 0, 1))), # model b's performance increasing in d
    rep(-0.5, 100))  # model c's performance constant
) %>%
  spread(model, loso_log_score)

preds <- runif(300, -5, 5)

test_that("objective works", {
  ## although the objective is only used for unit testing,
  ## need to make sure it's right!
  ## the implementation in stacking-estimation.R uses more numerically
  ## stable calculations, but is less intuitive.
  ## Here I implement the objective function again, less stably but more intuitively.
  
  obj_fn <- get_obj_fn(component_model_log_scores =
    as.matrix(
      loso_pred_res[, paste0("log_score_", component_models), drop = FALSE]
    ) %>%
      `storage.mode<-`("double"))
  
  manual_obj_fn <- function(preds, component_model_log_scores) {
    ## convert preds to matrix form with one row per observation and one column per component model
    preds <- preds_to_matrix(preds = preds, num_models = ncol(component_model_log_scores))
    
    temp <- exp(preds)
    denom <- matrix(rep(apply(temp, 1, sum), times = ncol(component_model_log_scores)),
      nrow = nrow(preds),
      ncol = ncol(preds))
    model_weights <- temp / denom
    
    scores <- exp(component_model_log_scores)
    
    temp <- apply(model_weights * scores, 1, sum)

    return(-1 * sum(log(temp)))
  }
  
  package_result <- obj_fn(preds)
  manual_result <- manual_obj_fn(preds = preds,
    component_model_log_scores =
      as.matrix(
        loso_pred_res[, paste0("log_score_", component_models), drop = FALSE]
      ) %>%
      `storage.mode<-`("double"))
  
  expect_equal(package_result, manual_result)
})


test_that("objective gradient works", {
  obj_fn <- get_obj_fn(component_model_log_scores =
    as.matrix(
      loso_pred_res[, paste0("log_score_", component_models), drop = FALSE]
    ) %>%
      `storage.mode<-`("double"))
  
  obj_deriv_fn <- get_obj_deriv_fn(component_model_log_scores =
    as.matrix(
      loso_pred_res[, paste0("log_score_", component_models), drop = FALSE]
    ) %>%
    `storage.mode<-`("double"))
  
  calc_obj_deriv <- obj_deriv_fn(preds)
  
  obj_fn_wrapper <- function(val, ind, preds) {
    preds[ind] <- val
    obj_fn(preds)
  }
  
  numeric_grad <- sapply(seq_along(preds), function(ind) {
    val <- preds[ind]
    as.numeric(attr(numericDeriv(quote(obj_fn_wrapper(val, ind, preds)), "val"), "gradient"))
  })
  
  expect_equal(calc_obj_deriv$grad, numeric_grad, tolerance = 10^-5)
})


test_that("objective gradient works", {
  obj_deriv_fn <- get_obj_deriv_fn(component_model_log_scores =
    as.matrix(
      loso_pred_res[, paste0("log_score_", component_models), drop = FALSE]
    ) %>%
    `storage.mode<-`("double"))
  
  calc_obj_deriv <- obj_deriv_fn(preds)
  
  obj_deriv_wrapper <- function(val, ind, preds) {
    preds[ind] <- val
    obj_deriv_fn(preds)$grad[ind]
  }
  
  numeric_grad <- sapply(seq_along(preds), function(ind) {
    val <- preds[ind]
    as.numeric(attr(numericDeriv(quote(obj_deriv_wrapper(val, ind, preds)), "val"), "gradient"))
  })
  
  expect_equal(calc_obj_deriv$hess, numeric_grad, tolerance = 10^-5)
})
