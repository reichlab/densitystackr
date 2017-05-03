### Estimate function that gives model weights based on observed inputs,
### using splines

#' Function to compute (log) weights of component models from the "predictions"
#' output by xgboost
#'
#' @param preds predictions from xgboost in matrix form, with num_models columns
#'   and num_obs rows
#' @param log boolean; return log of component model weights or just the weights
#'
#' @return a matrix with num_models columns and num_obs rows, with (log) weight
#'   for model m at observation i in entry [i, m]
compute_model_weights_from_spline_f_hats <- function(
  f_hats,
  M = M,
  J = J,
  log = FALSE) {
  if(J > 1) {
    preds <- matrix(nrow = nrow(f_hats), ncol = M)
    for(m in seq_len(M)) {
      preds[, m] <- apply(f_hats[, seq_len(J) + (m - 1) * J], 1, sum)
    }
  } else {
    preds <- f_hats
  }

  for(i in seq_len(nrow(preds))) {
    preds[i, ] <- preds[i, ] - logspace_sum(preds[i, ])
  }

  if(log) {
    return(preds)
  } else {
    return(exp(preds))
  }
}

#' Compute (log) weights of component models based on a densitystack fit and
#' new data.
#'
#' @param densitystack_fit a fit xgbstack object
#' @param newdata new x data
#' @param ntreelimit how many boosting iterations worth of trees to use
#' @param log boolean: return log of weights or original weights?
#'
#' @return (log) weights in the format determined by format
#'
#' @export
compute_model_weights <- function(
  density_stack_fit,
  newdata,
  log = FALSE) {
  if(!identical(class(density_stack_fit), "density_stack")) {
    stop("density_stack_fit must be an object of type density_stack!")
  }

  ## convert newdata to matrix format
  newdata_matrix <- Formula::model.part(density_stack_fit$formula,
    data = newdata,
    rhs = 1) %>%
      as.matrix() %>%
      `storage.mode<-`("double")

  ## get spline fitted values
  M <- ncol(density_stack_fit$model_scores)
  J <- ncol(density_stack_fit$dtrain)
  f_hat <- matrix(NA,
    nrow = nrow(newdata_matrix),
    ncol = M * J)
  m_j_pairs <- expand.grid(
    m = seq_len(M),
    j = seq_len(J),
    stringsAsFactors = FALSE
  )
  for(update_ind in seq_len(nrow(m_j_pairs))) {
    m <- m_j_pairs$m[update_ind]
    j <- m_j_pairs$j[update_ind]
    f_hat[, j + (m - 1) * J] <- predict(
      density_stack_fit$spline_fits[[update_ind]],
      x = newdata_matrix[, j])$y
  }

  ## convert to weights
  model_weights <-  compute_model_weights_from_spline_f_hats(f_hat,
    M = M,
    J = J,
    log = log)

  ## set column names
  colnames(model_weights) <-
    strsplit(as.character(density_stack_fit$formula)[2], " + ", fixed = TRUE)[[1]]

  ## return
  return(model_weights)
}

#' A factory-esque arrangement (not sure if there is an actual name for this
#' pattern) to manufacture an function to calculate first and second order
#' derivatives of the log-score objective, with needed quantities
#' accessible in its parent environment.  We do this becaues there's no way to
#' use the info attribute of the dtrain object to store the component model
#' log scores (as would be standard in the xgboost package).  But we need to
#' ensure that the objective function has access to these log scores when
#' called in parallel.
#'
#' @param component_model_log_scores N by M matrix where entry [i, m] has the
#'   log-score for observation i obtained from component model m
#'
#' @return a function that takes two arguments (preds and dtrain) and computes
#'   the first and second order derivatives of the log-score objective function
#'   for the stacked model.  i.e., it converts preds to model weights for each
#'   component model m at each observation i and then combines the component
#'   model log scores with those weights to obtain stacked model log scores.
#'   See the package vignette for derivations of these calculations.  This
#'   function is suitable for use as the "obj" function in a call to
#'   xgboost::xgb.train
compute_spline_working_weights_and_response <- function(
  f_hat_old,
  component_model_log_scores,
  dtrain,
  m,
  M,
  j,
  J,
  min_obs_weight = 1) {

  ## Compute log of component model weights at each observation
  log_weights <- compute_model_weights_from_spline_f_hats(f_hat_old,
    M = M,
    J = J,
    log = TRUE)

  ## adding log_weights and component_model_log_scores gets
  ## log(pi_mi) + log(f_m(y_i | x_i)) = log(pi_mi * f_m(y_i | x_i))
  ## in cell [i, m] of the result.  logspace sum the rows to get a vector with
  ## log(sum_m pi_mi * f_m(y_i | x_i)) in position i.
  log_weighted_scores <- log_weights + component_model_log_scores
  log_weighted_score_sums <- logspace_sum_matrix_rows(log_weighted_scores)

  ## calculate log of pi_{m} f_{m}(y_i | x_i) / f(y_i | x_i)
  log_prop_dens_from_m <- log_weighted_scores[, m] - log_weighted_score_sums

  ## calculate log of vector g from vignette, where
  ## g[i] = pi_{m} f_{m}(y_i | x_i) / f(y_i | x_i) - pi_{m}
  ## g[i] may be either positive or negative, so keep track of sign and log(|g|)
  temp <- cbind(log_prop_dens_from_m, log_weights[, m])
  sign_g <- sign(apply(temp, 1, function(temp_row) {temp_row[1] - temp_row[2]}))
  log_g <- vector(mode = "numeric", length = nrow(temp))
  pos_inds <- (sign_g > 0)
  if(sum(pos_inds) > 0) {
    log_g[pos_inds] <- logspace_sub_matrix_rows(temp[pos_inds, , drop = FALSE])
  }
  if(sum(!pos_inds) > 0) {
    log_g[!pos_inds] <-
      logspace_sub_matrix_rows(temp[!pos_inds, c(2, 1), drop = FALSE])
  }

  ## calculate log of observation weights from vignette, where
  ## w[i] = [ pi_{m} f_{m}(y_i | x_i) / f(y_i | x_i) ]^2 + pi_{m}
  ##            - [ pi_{m} f_{m}(y_i | x_i) / f(y_i | x_i) + pi_{m}^2 ]
  ## w[i] may be either positive or negative, so keep track of sign and log(|w|)
  term_1 <- logspace_sum_matrix_rows(
    cbind(
      2 * log_prop_dens_from_m,
      log_weights[, m]
    )
  )
  term_2 <- logspace_sum_matrix_rows(
    cbind(
      log_prop_dens_from_m,
      2 * log_weights[, m]
    )
  )
  temp <- cbind(term_1, term_2)
  sign_w <- sign(apply(temp, 1, function(temp_row) {temp_row[1] - temp_row[2]}))
  log_w <- vector(mode = "numeric", length = nrow(temp))
  pos_inds <- (sign_w > 0)
  if(sum(pos_inds) > 0) {
    log_w[pos_inds] <- logspace_sub_matrix_rows(temp[pos_inds, , drop = FALSE])
  }
  if(sum(!pos_inds) > 0) {
    log_w[!pos_inds] <- logspace_sub_matrix_rows(temp[!pos_inds, c(2, 1), drop = FALSE])
  }

  ## truncate below -- at this point, all sign_w are +1, so we ignore that
  log_w[(pos_inds & (log_w < log(min_obs_weight))) || !pos_inds] <-
    log(min_obs_weight)

  ## calculate working response, see vignette
  working_response <- sign_g * exp(log_g - log_w) + f_hat_old[, j + (m - 1) * J]

  ## return
  return(list(w = exp(log_w), y = working_response))
}

#' Fit a stacking model given a measure of performance for each component model
#' on a set of training data, and a set of covariates to use in forming
#' component model weights
#'
#' @param formula a formula describing the model fit.  left hand side should
#'   give columns in data with scores of models, separated by +.  right hand
#'   side should specify explanatory variables on which weights will depend.
#' @param data a data frame with variables in formula
#' @param booster what form of boosting to use? see xgboost documentation
#' @param subsample fraction of data to use in bagging.  not supported yet.
#' @param colsample_bytree fraction of explanatory variables to randomly select
#'   in growing each regression tree. see xgboost documentation
#' @param colsample_bylevel fraction of explanatory variables to randomly select
#'   in growing each level of the regression tree. see xgboost documentation
#' @param max_depth maximum depth of regression trees. see xgboost documentation
#' @param min_child_weight not recommended for use. see xgboost documentation
#' @param eta learning rate. see xgboost documentation
#' @param gamma Penalty on number of regression tree leafs. see xgboost documentation
#' @param lambda L2 regularization of contribution to model weights in each
#'   round. see xgboost documentation
#' @param alpha L1 regularization of contribution to model weights in each round.
#'   see xgboost documentation
#' @param nrounds see xgboost documentation
#' @param cv_params optional named list of parameter values to evaluate loss
#'   via cross-validation. Each component is a vector of parameter values with
#'   name one of "booster", "subsample", "colsample_bytree",
#'   "colsample_bylevel", "max_depth", "min_child_weight", "eta", "gamma",
#'    "lambda", "alpha", "nrounds"
#' @param cv_folds list specifying observation groups to use in cross-validation
#'   each list component is a numeric vector of observation indices.
#' @param cv_nfolds integer specifying the number of cross-validation folds to
#'   use.  if cv_folds was provided, cv_nfolds is ignored.  if cv_folds was not
#'   provided, the data will be randomly partitioned into cv_nfolds groups
#' @param cv_refit character describing which of the models specified by the
#'   values in cv_params to refit using the full data set. Either "best",
#'   "ttest", or "none".
#' @param update an object of class xgbstack to update
#' @param nthread number of threads to use
#' @param verbose how much output to generate along the way. 0 for no logging,
#'   1 for some logging
#'
#' @return an estimated xgbstack object, which contains a gradient tree boosted
#'   fit mapping observed variables to component model weights
#'
#' @export
density_stack_splines_fixed_lambda <- function(formula,
  data,
  lambda,
  min_obs_weight = 1,
  tol = 10^{-5},
  maxit = 10^5,
  verbose = 0) {
  formula <- Formula::Formula(formula)

  ## response, as a matrix of type double
  model_scores <- Formula::model.part(formula, data = data, lhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")

  ## predictors, in format used in xgboost
  dtrain <- Formula::model.part(formula, data = data, rhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")

  ## number of models, observations, covariates
  M <- ncol(model_scores)
  N <- nrow(dtrain)
  J <- ncol(dtrain)

  ## clean up lambda argument
  lambda <- as.numeric(lambda)
  if(identical(length(lambda), 1L)) {
    lambda <- rep(lambda, J)
  } else if(!identical(length(lambda))) {
    stop("lambda must be a numeric vector of length 1 or J where J is the number of predictive covariates")
  }

  ## get fit
  f_hat_new <- matrix(0, nrow = N, ncol = M * J)
  f_hat_ssd <- tol + 1
  m_j_pairs <- expand.grid(
    m = seq_len(M),
    j = seq_len(J),
    stringsAsFactors = FALSE
  )
  it_ind <- 1L
  while(f_hat_ssd > tol && it_ind <= maxit) {
    ## store spline fit values from previous iteration
    f_hat_old <- f_hat_new

    for(update_ind in seq_len(nrow(m_j_pairs))) { # SWITCH TO FOREACH LATER??
      m <- m_j_pairs$m[update_ind]
      j <- m_j_pairs$j[update_ind]

      ## Fit spline for model m, covariates j
      temp <- compute_spline_working_weights_and_response(
        f_hat_old = f_hat_old,
        component_model_log_scores = model_scores,
        dtrain = dtrain,
        m = m,
        M = M,
        j = j,
        J = J,
        min_obs_weight = min_obs_weight)

      spline_fit_m_j <- smooth.spline(
        x = dtrain[, j],
        y = temp$y,
        w = temp$w,
        lambda = lambda[j],
        cv = NA,
        keep.data = TRUE
      )

      ## update f_hat_new
      fitted_values <- data.frame(
        x = spline_fit_m_j$x,
        y = spline_fit_m_j$y
      )
      f_hat_new[, j + (m - 1) * J] <-
        data.frame(x = dtrain[, j]) %>%
          dplyr::left_join(fitted_values, by = "x") %>%
          `[[`("y")
    }

    ## sum of squared differences in fitted spline values between iterations
    ## comparison to tol is stopping criterion
    f_hat_ssd <- sum((f_hat_new - f_hat_old)^2)

    if(verbose >= 1) {
      cat(paste0("Iteration ", it_ind, "; f_hat_ssd = ", f_hat_ssd, "\n"))
    }

    ## update iteration index
    ## comparison to maxit is stopping criterion
    it_ind <- it_ind + 1L
  }

  if(it_ind == maxit) {
    warning("Reached maximum number of iterations without convergence.")
  }

  ## one last update to get final spline fits to save/return
  spline_fits <- lapply(seq_len(nrow(m_j_pairs)),
    function(update_ind) {
      m <- m_j_pairs$m[update_ind]
      j <- m_j_pairs$j[update_ind]

      ## Fit spline for model m, covariates j
      temp <- compute_spline_working_weights_and_response(
        f_hat_old = f_hat_old,
        component_model_log_scores = model_scores,
        dtrain = dtrain,
        m = m,
        M = M,
        j = j,
        J = J,
        min_obs_weight = min_obs_weight)

      spline_fit_m_j <- smooth.spline(
        x = dtrain[, j],
        y = temp$y,
        w = temp$w,
        lambda = lambda[j],
        cv = NA,
        keep.data = FALSE
      )
      return(spline_fit_m_j)
  })

  ## return
  return(structure(
    list(spline_fits = spline_fits,
      formula = formula,
      model_scores = model_scores,
      dtrain = dtrain,
      lambda = lambda),
    class = "density_stack"
  ))
}
