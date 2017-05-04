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

log_lik_score <- function(log_model_weights, log_model_scores) {
  ## adding log_weights and component_model_log_scores gets
  ## log(pi_mi) + log(f_m(y_i | x_i)) = log(pi_mi * f_m(y_i | x_i))
  ## in cell [i, m] of the result.  logspace sum the rows to get a vector with
  ## log(sum_m pi_mi * f_m(y_i | x_i)) in position i.
  ## sum that vector to get the objective.
  return(sum(logspace_sum_matrix_rows(log_model_weights + log_model_scores)))
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
density_stack_splines_fixed_regularization <- function(formula,
  data,
  df = NULL,
  spar = NULL,
  lambda = NULL,
  min_obs_weight = 1,
  tol = 10^{-5},
  maxit = 10^5,
  verbose = 0) {
  formula <- Formula::Formula(formula)

  ## response, as a matrix of type double
  model_scores <- Formula::model.part(formula, data = data, lhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")

  ## predictors, as a matrix of type double
  dtrain <- Formula::model.part(formula, data = data, rhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")

  ## number of models, observations, covariates
  M <- ncol(model_scores)
  N <- nrow(dtrain)
  J <- ncol(dtrain)

  ## list of arguments to smooth.spline(), suitable for use with do.call()
  ## dtrain and temp are objects that will have been intantiated at the time
  ## smooth.spline() is called below.
  smooth.spline_args <- list(
    x = quote(dtrain[, j]),
    y = quote(temp$y),
    w = quote(temp$w),
    cv = NA,
    keep.data = FALSE
  )

  ## clean up regularization parameter and add to smooth.spline_args
  if(!missing(df) && !is.null(df)) {
    df <- as.numeric(df)
    if(identical(length(df), 1L)) {
      df <- rep(df, J)
      smooth.spline_args <- c(smooth.spline_args,
        df = quote(df[j]))
      smooth.spline_args$cv <- FALSE
    } else if(!identical(length(df), J)) {
      stop("If provided, df must be a numeric vector of length 1 or J where J is the number of predictive covariates.")
    }
  } else if(!missing(lambda) && !is.null(lambda)) {
    lambda <- as.numeric(lambda)
    if(identical(length(lambda), 1L)) {
      lambda <- rep(lambda, J)
      smooth.spline_args <- c(smooth.spline_args,
        lambda = quote(lambda[j]))
    } else if(!identical(length(lambda), J)) {
      stop("If provided, lambda must be a numeric vector of length 1 or J where J is the number of predictive covariates.")
    }
  } else {
    if(missing(spar) || is.null(spar)) {
      warning("No regularization parameters included: setting spar = 0.5!")
      spar <- 0.5
    } else {
      spar <- as.numeric(spar)
    }

    if(identical(length(spar), 1L)) {
      spar <- rep(spar, J)
      smooth.spline_args <- c(smooth.spline_args,
        spar = quote(spar[j]))
    } else if(!identical(length(spar), J)) {
      stop("If provided, spar must be a numeric vector of length 1 or J where J is the number of predictive covariates.")
    }
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

      spline_fit_m_j <- do.call("smooth.spline", smooth.spline_args)

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
      message(paste0("Iteration ", it_ind, "; f_hat_ssd = ", f_hat_ssd, "\n"))
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

      spline_fit_m_j <- do.call("smooth.spline", smooth.spline_args)

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



#' Fit a stacking model given a measure of performance for each component model
#' on a set of training data, and a set of covariates to use in forming
#' component model weights.  Optionally, use cross-validation to select
#' amount of regularization.
#'
#' @export
density_stack <- function(formula,
  data,
  df = NULL,
  spar = NULL,
  lambda = NULL,
  min_obs_weight = 0,
  tol = 10^{-5},
  maxit = 10^5,
  cv_folds = NULL,
  cv_nfolds = 10L,
  verbose = 0) {

  ## keep only one set of regularization parameters
  ## prefer spar, then lamba, then df
  if(!missing(spar) && !is.null(spar)) {
    if(!missing(df) || !missing(lambda)) {
      warning("Multiple sets of regularization parameters provided -- using only spar")
      df <- NULL
      lambda <- NULL
    }
  } else if(!missing(lambda) && !is.null(lambda)) {
    if(!missing(df)) {
      warning("Multiple sets of regularization parameters provided -- using only lambda")
      df <- NULL
    }
  } else if(missing(df)) {
    warning("No regularization parameter provided -- setting spar = 0.5!")
    spar <- 0.5
  }

  ## list of arguments suitable for use in a call to via do.call()
  density_stack_fixed_reg_params <- list(
    formula = formula,
    min_obs_weight = min_obs_weight,
    tol = tol,
    maxit = maxit,
    verbose = verbose - 1L)

  if(ifelse(is.data.frame(df), nrow(df) > 1, FALSE) ||
    ifelse(is.data.frame(spar), nrow(spar) > 1, FALSE) ||
    ifelse(is.data.frame(lambda), nrow(lambda) > 1, FALSE)) {
    ## cross-validation for regularization parameter selection

    ## if they weren't provided, get sets of observations for cv folds
    ## otherwise, set cv_nfolds = number of folds provided
    if(is.null(cv_folds)) {
      cv_fold_memberships <- cut(seq_len(nrow(data)), cv_nfolds) %>%
        as.numeric() %>%
        sample(size = nrow(data), replace = FALSE)
      cv_folds <- lapply(seq_len(cv_nfolds),
        function(fold_ind) {
          which(cv_fold_memberships == fold_ind)
        }
      )
    } else {
      cv_nfolds <- length(cv_folds)
    }

    ## get data, names of covariates
    formula <- Formula::Formula(formula)

    ## response, as a matrix of type double
    model_scores <- Formula::model.part(formula, data = data, lhs = 1) %>%
      as.matrix() %>%
      `storage.mode<-`("double")

    ## predictors, as a matrix of type double
    dtrain <- Formula::model.part(formula, data = data[1, , drop = FALSE], rhs = 1) %>%
      as.matrix() %>%
      `storage.mode<-`("double")

    ## number of models, observations, covariates
    M <- ncol(model_scores)
    N <- nrow(model_scores)
    J <- ncol(dtrain)
    covar_names <- colnames(dtrain)

    ## ensure that either
    ## (a) regularization parameter values were provided for all covariates, or
    ## (b) only one set of regularization parameter values was provided,
    ##     to be used for all covariates
    if(!is.null(df)) {
      if(ncol(df) > 1) {
        if(!all(covar_names %in% colnames(df))) {
          stop("Cross-validation parameter grid must have only one column or include values for all covariates in right-hand side of formula: ", paste(covar_names, sep = ", "))
        }
        ## pull out only relevant parameter values, in correct order
        df <- df[, covar_names]
      }

      ## add argument to density_stack_fixed_reg_params
      density_stack_fixed_reg_params$df <- quote(df[cv_ind, ])
    } else if(!is.null(spar)) {
      if(ncol(spar) > 1) {
        if(!all(covar_names %in% colnames(spar))) {
          stop("Cross-validation parameter grid must have only one column or include values for all covariates in right-hand side of formula: ", paste(covar_names, sep = ", "))
        }
        ## pull out only relevant parameter values, in correct order
        spar <- spar[, covar_names]
      }

      ## add argument to density_stack_fixed_reg_params
      density_stack_fixed_reg_params$spar <- quote(spar[cv_ind, ])
    } else if(!is.null(lambda)) {
      if(ncol(lambda) > 1) {
        if(!all(covar_names %in% colnames(lambda))) {
          stop("Cross-validation parameter grid must have only one column or include values for all covariates in right-hand side of formula: ", paste(covar_names, sep = ", "))
        }
        ## pull out only relevant parameter values, in correct order
        lambda <- lambda[, covar_names]
      }

      ## add argument to density_stack_fixed_reg_params
      density_stack_fixed_reg_params$lambda <- quote(lambda[cv_ind, ])
    }

    ## construct data frame with all combinations of parameter value settings
    cv_results <- expand.grid(c(df, spar, lambda), stringsAsFactors = FALSE)
    num_reg_param_cols <- ncol(cv_results)

    # ## if update was provided, subset to only rows specifying new parameter
    # ## combinations (don't re-estimate cv results already in update)
    # if(update_same_data_and_cv) {
    #   cv_results <- cv_results %>%
    #     anti_join(update$cv_results, by = names(base_params)[names(base_params) != "nthread"])
    # }

    ## space for cv results
    cv_results <- cbind(cv_results,
      matrix(NA, nrow = nrow(cv_results), ncol = cv_nfolds + 1) %>%
        `colnames<-`(
          c(paste0("cv_log_score_fold_", seq_len(cv_nfolds)), "cv_log_score_mean")
        )
    )

    density_stack_fixed_reg_params$data <-
      quote(data[-cv_folds[[k]], , drop = FALSE])

    for(cv_ind in seq_len(nrow(cv_results))) {
      if(verbose >= 1) {
        message(paste0("Fitting cv model ",
          cv_ind,
          " of ",
          nrow(cv_results)
        ))
      }

      ## for each k = 1, ..., cv_nfolds,
      ##  a) get fit leaving out fold k
      ##  b) get log score for fold k (possibly for multiple values of nrounds)
      cv_results[cv_ind, paste0("cv_log_score_fold_", seq_len(cv_nfolds))] <-
        foreach(k = seq_len(cv_nfolds), .combine = c) %dopar% {
  #      for(k in seq_len(cv_nfolds)) { # REPLACE WITH FOREACH LATER
          ## step a) get fit leaving out fold k
          stacking_fit_k <- do.call("density_stack_splines_fixed_regularization",
            density_stack_fixed_reg_params)

          ## step b) get log score for fold k (val for validation)
            log_lik_score(
              log_model_weights = compute_model_weights(stacking_fit_k,
                newdata = data[cv_folds[[k]], covar_names, drop = FALSE],
                log = TRUE),
              log_model_scores = model_scores[cv_folds[[k]], , drop = FALSE])
        } # end code to get log score for each k-fold (foreach)
    } # end code to get log score for each parameter combination

    cv_results$cv_log_score_mean <-
      apply(cv_results[, paste0("cv_log_score_fold_", seq_len(cv_nfolds))],
        1,
        mean)

    # ## if update was provided, merge cv_results with previous cv_results
    # if(update_same_data_and_cv) {
    #   cv_results <- cv_results %>%
    #     bind_rows(update$cv_results)
    # }

    ## get fit with all training data based on selected parameters
    density_stack_fixed_reg_params$data <- quote(data)

    best_params_ind <- which.max(cv_results$cv_log_score_mean)
    reg_param_name <- names(density_stack_fixed_reg_params)[
      names(density_stack_fixed_reg_params) %in% c("df", "spar", "lambda")
    ]
    density_stack_fixed_reg_params[[reg_param_name]] <-
      quote(cv_results[best_params_ind, seq_len(num_reg_param_cols)])

    density_stack_fit <- do.call("density_stack_splines_fixed_regularization",
      density_stack_fixed_reg_params)

    density_stack_fit$cv_results <- cv_results
  } else {
    ## no cross-validation for parameter selection
    ## get fit with all training data based on provided parameters
    density_stack_fit <- density_stack_splines_fixed_regularization(
      formula = formula,
      data = data,
      df = df,
      spar = spar,
      lambda = lambda,
      min_obs_weight = min_obs_weight,
      tol = tol,
      maxit = maxit,
      verbose = verbose)
  }

  return(density_stack_fit)
}
