### Estimate function that gives model weights based on observed inputs

#' Convert vector of predictions to matrix format
#' From line 56 at https://github.com/dmlc/xgboost/blob/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/src/objective/multiclass_obj.cc,
#' it appears that preds is stored in column-major order with
#' observations in columns and classes/models in rows
#' i.e., preds[k * nclass + i] is prediction for model i at index k
#' 
#' @param preds vector of predictions as obtained from xgb.train
#' @param number of models
#' 
#' @return preds in matrix form, with num_models columns and num_obs rows
preds_to_matrix <- function(preds, num_models) {
  num_obs <- length(preds) / num_models
  dim(preds) <- c(num_models, num_obs)
  return(t(preds))
}

#' Function to compute (log) weights of component models from the "predictions"
#' output by xgboost
#' 
#' @param preds predictions from xgboost in matrix form, with num_models columns
#'   and num_obs rows
#' @param log boolean; return log of component model weights or just the weights
#' 
#' @return a matrix with num_models columns and num_obs rows, with (log) weight
#'   for model m at observation i in entry [i, m]
compute_model_weights_from_preds <- function(preds, log = FALSE) {
  for(i in seq_len(nrow(preds))) {
    preds[i, ] <- preds[i, ] - logspace_sum(preds[i, ])
  }
  
  if(log) {
    return(preds)
  } else {
    return(exp(preds))
  }
}

#' Compute (log) weights of component models based on an xgbstack fit and
#' new data.
#' 
#' Format "complete" is only partially implemented.
#' 
#' @param xgbstack_fit a fit xgbstack object
#' @param newdata new x data
#' @param ntreelimit how many boosting iterations worth of trees to use
#' @param log boolean: return log of weights or original weights?
#' @param format string, either "bare" to return an nrow(newdata) by num_models
#'   matrix of (log) weights, or "complete" to return a data frame with
#'   nrow(newdata) times num_models data frame with both weights and log_weights
#'   as well as parameter values and newdata
#' 
#' @return (log) weights in the format determined by format
#' 
#' @export
compute_model_weights <- function(
  xgbstack_fit,
  newdata,
  ntreelimit,
  log = FALSE,
  format = "bare") {
  if(!identical(class(xgbstack_fit), "xgbstack")) {
    stop("xgbstack_fit must be an object of type xgbstack!")
  }
  
  if(identical(class(xgbstack_fit$fit), "raw")) {
    ## fit is just one "raw" representation of a fit for a single parameter set
    ## get weights from just that fit
    
    ## convert newdata to format used in xgboost
    newdata_matrix <- Formula::model.part(xgbstack_fit$formula, data = newdata, rhs = 1) %>%
      as.matrix() %>%
      `storage.mode<-`("double")
    newdata <- xgb.DMatrix(data = newdata_matrix)
    
    ## get something proportional to log(weights)
    xgb_fit <- xgb.load(xgbstack_fit$fit)
    if(missing(ntreelimit)) {
      preds <- predict(xgb_fit, newdata = newdata)
    } else {
      preds <- predict(xgb_fit, ntreelimit = ntreelimit, newdata = newdata)
    }
    
    ## convert to weights
    preds <- preds_to_matrix(preds, num_models = xgbstack_fit$num_models)
    model_weights <- compute_model_weights_from_preds(preds, log = log)
    
    ## set column names
    colnames(model_weights) <-
      strsplit(as.character(xgbstack_fit$formula)[2], " + ", fixed = TRUE)[[1]]
    
    if(identical(format, "complete")) {
      stop("format complete is only partially implemented")
      ## need to add params and newdata and gather current colnames(model_weights)
    }
  } else {
    ## fit is a list of "raw" representations of fits for parameter sets in rows
    ## of xgbstack_fit$params_refit
    ## get weights from inidivual parameter sets in xgbstack_fit$params, and
    ## a combination of those (on log scale)
    
    ## convert newdata to format used in xgboost
    newdata_matrix <- Formula::model.part(xgbstack_fit$formula, data = newdata, rhs = 1) %>%
      as.matrix() %>%
      `storage.mode<-`("double")
    newdata <- xgb.DMatrix(data = newdata_matrix)
    
    ## template with unique parameter sets for which we need to obtain weights
    weights_template <- as.data.frame(xgbstack_fit$params)
    
    ## colnames of result are based on response variable names:
    ## weights are assigned to models whose names are presumably encoded in the
    ## response variable names
    response_var_names <-
      strsplit(as.character(xgbstack_fit$formula)[2], " + ", fixed = TRUE)[[1]]
    
    ## weights matrix that we will populate below
    num_param_sets <- nrow(weights_template)
    num_models <- length(response_var_names)
    weights_combined <- matrix(NA, nrow = nrow(newdata_matrix), ncol = num_param_sets * num_models)
    colnames(weights_combined) <- as.vector(outer(response_var_names, seq_len(num_param_sets),
      function(response_var_name, param_set_ind) {
        paste0(
          response_var_name,
          "_params_ind_",
          param_set_ind
        )
      }))
    
    ## for each parameter set in weights_template, get corresponding weights
    params_to_match_on <- c("booster", "subsample", "colsample_bytree",
      "colsample_bylevel", "max_depth", "min_child_weight", "eta", "gamma",
      "lambda", "alpha")
    for(weights_ind in seq_len(nrow(weights_template))) {
      message(paste0("param set ", weights_ind, " of ", nrow(weights_template), "\n"))
      ## get the index of the fit to use: all params other than nrounds match
      ## corresponding values in predictions_combined
      fit_ind <- which(sapply(seq_len(nrow(xgbstack_fit$params_refit)),
        function(refit_ind) {
          all(weights_template[weights_ind, params_to_match_on] ==
              xgbstack_fit$params_refit[refit_ind, params_to_match_on])
        }))
      
      xgb_fit <- xgb.load(xgbstack_fit$fit[[fit_ind]])
      preds <- predict(xgb_fit,
        ntreelimit = weights_template$nrounds[weights_ind],
        newdata = newdata)
      
      ## convert to weights
      preds <- preds_to_matrix(preds, num_models = xgbstack_fit$num_models)
      single_model_weights <- compute_model_weights_from_preds(preds, log = FALSE)
      
      ## add to combined weights data frame
      inds <- seq(from = (weights_ind - 1) * num_models + 1, length = num_models)
      weights_combined[, inds] <- single_model_weights
    }
    
    ## get combined weights across all parameter sets, averaging on non-log scale
    mean_weights_matrix <- matrix(
      rep(diag(num_models) / num_param_sets, times = num_param_sets),
      nrow = num_models * num_param_sets,
      ncol = num_models,
      byrow = TRUE)
    model_weights <- cbind(
      weights_combined,
      (weights_combined %*% mean_weights_matrix) %>%
        `colnames<-`(paste0(
          response_var_names,
          "_params_combined"
        ))
    )
    if(log) {
      model_weights <- log(model_weights)
    }
    
    if(identical(format, "complete")) {
      stop("format \"complete\" not yet implemented")
    }
  } # end conditioning on class(xgbstack_fit$fit)
  
  ## return
  return(model_weights)
}

#' A factory-esque arrangement (not sure if there is an actual name for this
#' pattern) to manufacture an objective function with needed quantities
#' accessible in its parent environment.  We do this because there's no way to
#' use the info attribute of the dtrain object to store the component model
#' log scores (as would be standard in the xgboost package).  But we need to
#' ensure that the objective function has access to these log scores when
#' called in parallel.
#' 
#' @param component_model_log_scores N by M matrix where entry [i, m] has the
#'   log-score for observation i obtained from component model m
#' 
#' @return a function that takes two arguments (preds and dtrain) and computes
#'   the log-score objective function for the stacked model.  i.e., it converts
#'   preds to model weights for each component model m at each observation i
#'   and then combines the component model log scores with those weights to
#'   obtain stacked model log scores
get_obj_fn <- function(component_model_log_scores) {
  ## evaluate arguments so that they're not just empty promises
  component_model_log_scores
  
  ## create function to calculate objective
  obj_fn <- function(preds, dtrain) {
    ## convert preds to matrix form with one row per observation and one column per component model
    preds <- preds_to_matrix(preds = preds, num_models = ncol(component_model_log_scores))
    
    ## Compute log of component model weights at each observation
    log_weights <- compute_model_weights_from_preds(preds, log = TRUE)
    
    ## adding log_weights and component_model_log_scores gets
    ## log(pi_mi) + log(f_m(y_i | x_i)) = log(pi_mi * f_m(y_i | x_i))
    ## in cell [i, m] of the result.  logspace sum the rows to get a vector with
    ## log(sum_m pi_mi * f_m(y_i | x_i)) in position i.
    ## sum that vector to get the objective.
    return(-1 * sum(logspace_sum_matrix_rows(log_weights + component_model_log_scores)))
  }
  
  ## return function to calculate objective
  return(obj_fn)
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
get_obj_deriv_fn <- function(component_model_log_scores, dtrain_Rmatrix) {
  ## evaluate arguments so that they're not just empty promises
  component_model_log_scores
  if(!missing(dtrain_Rmatrix)) {
    dtrain_Rmatrix
  }
  
  ## create function to calculate objective
  obj_deriv_fn <- function(preds, dtrain) {
    ## convert preds to matrix form with one row per observation and one column per component model
    preds <- preds_to_matrix(preds = preds, num_models = ncol(component_model_log_scores))
    
    ## Compute log of component model weights at each observation
    log_weights <- compute_model_weights_from_preds(preds, log = TRUE)
    
    ## adding preds and component_model_log_scores gets
    ## log(pi_mi) + log(f_m(y_i | x_i)) = log(pi_mi * f_m(y_i | x_i))
    ## in cell [i, m] of the result.  logspace sum the rows to get a vector with
    ## log(sum_m pi_mi * f_m(y_i | x_i)) in position i.
    ## sum that vector to get the objective.
    log_weighted_scores <- log_weights + component_model_log_scores
    log_weighted_score_sums <- logspace_sum_matrix_rows(log_weighted_scores)
    
    ## calculate gradient
    ## is there a way to do exponentiation last in the lines below,
    ## instead of in the calculations of term1 and term2?
    ## think i may need to vectorize logspace_sub?
    grad_term1 <- exp(sweep(log_weighted_scores, 1, log_weighted_score_sums, `-`))
    grad_term2 <- exp(log_weights)
    grad <- grad_term1 - grad_term2
    grad <- as.vector(t(grad))
    
    ## calculate hessian
    hess <- grad_term1 - grad_term1^2 - grad_term2 + grad_term2^2
    hess <- as.vector(t(hess))
    
    ## return
    return(list(grad = -1 * grad, hess = -1 * hess))
  }
  
  ## return function to calculate derivatives of objective
  return(obj_deriv_fn)
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
xgbstack <- function(formula,
  data,
  booster = "gbtree",
  subsample = 1,
  colsample_bytree = 1,
  colsample_bylevel = 1,
  max_depth = 6,
  min_child_weight = -10^10,
  eta = 0.3,
  gamma = 0,
  lambda = 0,
  alpha = 0,
  nrounds = 10,
  cv_params = NULL,
  cv_folds = NULL,
  cv_nfolds = 10L,
  cv_refit = "ttest",
  update = NULL,
  nthread = NULL,
  verbose = 0) {
  formula <- Formula::Formula(formula)
  
  ## response, as a matrix of type double
  model_scores <- Formula::model.part(formula, data = data, lhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")
  
  ## predictors, in format used in xgboost
  dtrain_Rmatrix <- Formula::model.part(formula, data = data, rhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")
  dtrain <- xgb.DMatrix(
    data = dtrain_Rmatrix
  )
  
  ## process the update argument
  update_same_data_and_cv <- FALSE
  if(!is.null(update)) {
    if(!identical(class(update, "xgbstack"))) {
      stop("Argument update must be an object of class xgbstack.")
    }
    
    if(identical(formula, update$formula) &&
        identical(model_scores, update$model_scores) &&
        identical(dtrain_Rmatrix, update$dtrain_Rmatrix)) {
      ## data from update is same as data provided now
      if(!is.null(cv_params)) {
        ## was provided cv_folds/cv_nfolds same as update$cv_folds?
        if(!missing(cv_folds) && !is.null(cv_folds)) {
          if(identical(cv_folds, update$cv_folds)) {
            update_same_data_and_cv <- TRUE
            cv_nfolds <- length(cv_folds)
          }
        } else if(!missing(cv_nfolds) && !is.null(cv_nfolds)) {
          if(identical(as.integer(cv_nfolds, length(update$cv_folds)))) {
            update_same_data_and_cv <- TRUE
            cv_folds <- update$cv_folds
          }
        } else {
          ## did not provide cv_folds or cv_nfolds, reuse values from update
          update_same_data_and_cv <- TRUE
          cv_folds <- update$cv_folds
          cv_nfolds <- length(update$cv_folds)
        }
      }
    }
  }
  
  ## base parameter values -- defaults or specified by arguments to this function
  ## if cv_params were provided, base_params will be overridden by cv_params
  base_params <- list(
    booster = booster,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    colsample_bylevel = colsample_bylevel,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    eta = eta,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    nrounds = nrounds,
    num_class = ncol(model_scores)
  )
  if(!missing(nthread) && !is.null(nthread)) {
    base_params$nthread <- as.integer(nthread)
  }
  
  ## function to calculate grad and hess of objective function based on
  ## full data set.  used in fitting gradient tree boosted models below
  obj_deriv_fn <- get_obj_deriv_fn(
    component_model_log_scores = model_scores,
    dtrain_Rmatrix = dtrain_Rmatrix)
  
  
  if(is.null(cv_params)) {
    ## no cross-validation for parameter selection
    ## use base_params
    params <- base_params
    
    ## get fit with all training data based on provided parameters
    fit <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      obj = obj_deriv_fn,
      verbose = 0
    ) %>%
      xgb.save.raw()
  } else {
    ## estimation of some parameters via cross validation was specified
    
    ## if they weren't provided, get sets of observations for cv folds
    ## otherwise, set cv_nfolds = number of folds provided
    if(is.null(cv_folds)) {
      cv_fold_memberships <- cut(seq_len(nrow(dtrain_Rmatrix)), cv_nfolds) %>%
        as.numeric() %>%
        sample(size = nrow(dtrain_Rmatrix), replace = FALSE)
      cv_folds <- lapply(seq_len(cv_nfolds),
        function(fold_ind) {
          which(cv_fold_memberships == fold_ind)
        }
      )
    } else {
      cv_nfolds <- length(cv_folds)
    }
    
    ## construct data frame with all combinations of parameter value settings
    combined_params <- c(
      base_params[!(names(base_params) %in% names(cv_params))],
      cv_params)[names(base_params)]
    cv_results <- expand.grid(combined_params, stringsAsFactors = FALSE)
    
    ## if update was provided, subset to only rows specifying new parameter
    ## combinations (don't re-estimate cv results already in update)
    if(update_same_data_and_cv) {
      cv_results <- cv_results %>%
        anti_join(update$cv_results, by = names(base_params))
    }
    
    ## space for cv results
    cv_results <- cbind(cv_results,
      matrix(NA, nrow = nrow(cv_results), ncol = cv_nfolds + 1) %>%
        `colnames<-`(
          c(paste0("cv_log_score_fold_", seq_len(cv_nfolds)), "cv_log_score_mean")
        )
    )
    
    ## if nrounds is a parameter to choose by cross-validation,
    ## only fit the models with largest number of nrounds,
    ## then get predictions with fewer rounds by using ntreelimit argument
    if("nrounds" %in% names(cv_params)) {
      model_inds_to_fit <- which(cv_results$nrounds == max(cv_params$nrounds))
    } else {
      model_inds_to_fit <- seq_len(nrow(cv_results))
    }
    
    for(cv_ind in model_inds_to_fit) {
      params <- cv_results[cv_ind, ]
      attr(params, "out.attrs") <- NULL
      params <- as.list(params)
      params <- c(params,
        base_params[!(names(base_params) %in% names(params))])
      
      if(verbose >= 1) {
        message(paste0("Fitting cv model ",
          which(model_inds_to_fit == cv_ind),
          " of ",
          length(model_inds_to_fit),
          ".  Parameters are: ",
          paste0(names(unlist(params[seq_along(base_params)])),
            " = ",
            unlist(params[seq_along(base_params)]),
            collapse = "; ")
        ))
      }
      
      ## get all rows for parameter combinations where everything other than
      ## nrounds matches what is in the current set of parameters
      if("nrounds" %in% names(cv_params)) {
        if(length(cv_params) == 1) {
          similar_param_rows <- seq_len(nrow(cv_results))
        } else {
          cols_to_examine <- which(names(base_params) != "nrounds")
          similar_param_rows <- which(sapply(seq_len(nrow(cv_results)),
            function(possible_ind) {
              all(cv_results[possible_ind, cols_to_examine] == cv_results[cv_ind, cols_to_examine], na.rm = TRUE)
            }
          ))
        }
      } else {
        similar_param_rows <- cv_ind
      }
      
      ## for each k = 1, ..., cv_nfolds,
      ##  a) get xgb fit leaving out fold k
      ##  b) get log score for fold k (possibly for multiple values of nrounds)
      for(k in seq_len(cv_nfolds)) {
        ## step a) get xgb fit leaving out fold k
        dtrain_Rmatrix_k <- dtrain_Rmatrix[-cv_folds[[k]], , drop = FALSE]
        dtrain_k <- xgb.DMatrix(
          data = dtrain_Rmatrix_k
        )
        
        obj_deriv_fn_train_k <- get_obj_deriv_fn(
          component_model_log_scores = model_scores[-cv_folds[[k]], , drop = FALSE],
          dtrain_Rmatrix = dtrain_Rmatrix_k)
        
        fit_k <- xgb.train(
          params = params,
          data = dtrain_k,
          nrounds = params$nrounds,
          obj = obj_deriv_fn_train_k,
          verbose = 0
        )
        
        ## step b) get log score for fold k (val for validation)
        dval_Rmatrix_k <- dtrain_Rmatrix[cv_folds[[k]], , drop = FALSE]
        dval_k <- xgb.DMatrix(
          data = dval_Rmatrix_k
        )
        
        obj_fn_val_k <- get_obj_fn(
          component_model_log_scores = model_scores[cv_folds[[k]], , drop = FALSE])
        
        ## obj_fn returns -1* log score (thing to minimize)
        ## to avoid confusion (?) I'll return log score
        for(row_to_eval in similar_param_rows) {
          cv_results[row_to_eval, paste0("cv_log_score_fold_", k)] <-
            -1 * obj_fn_val_k(
              preds = predict(fit_k,
                newdata = dval_k,
                ntreelimit = cv_results[row_to_eval, "nrounds"])
            )
        }
      } # end code to get log score for each k-fold
    } # end code to get log score for each parameter combination
    
    cv_results$cv_log_score_mean <- apply(cv_results[, paste0("cv_log_score_fold_", seq_len(cv_nfolds))], 1, mean)
#    cv_results$cv_log_score_sd <- apply(cv_results[, paste0("cv_log_score_fold_", seq_len(cv_nfolds))], 1, sd)
    
    ## if update was provided, merge cv_results with previous cv_results
    if(update_same_data_and_cv) {
      cv_results <- cv_results %>%
        bind_rows(update$cv_results)
    }
    
    ## get fit with all training data based on selected parameters
    if(identical(cv_refit, "best")) {
      ## fit based on the single set of parameter values with best performance
      ## best ind has highest log score
      ## (see comment about multiplication by -1 above)
      best_params_ind <- which.max(cv_results$cv_log_score_mean)
      
      params <- cv_results[best_params_ind, seq_len(ncol(cv_results) - cv_nfolds - 1)]
      attr(params, "out.attrs") <- NULL
      params <- as.list(params)
      params <- c(params,
        base_params[!(names(base_params) %in% names(params))])
      
      fit <- xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        obj = obj_deriv_fn,
        verbose = 0
      ) %>%
        xgb.save.raw()
    } else if(identical(cv_refit, "ttest")) {
      ## fits based on all sets of parameter values with cv performance
      ## not different from the cv performance of the best parameter set
      ## according to a paired t test looking at results from all cv folds
      ## best ind has highest log score
      ## (see comment about multiplication by -1 above)
      best_params_ind <- which.max(cv_results$cv_log_score_mean)
      refit_params_inds <- sapply(seq_len(nrow(cv_results)),
        function(ind) {
          t.test(
            x = as.numeric(cv_results[ind, paste0("cv_log_score_fold_", seq_len(cv_nfolds))]),
            y = as.numeric(cv_results[best_params_ind, paste0("cv_log_score_fold_", seq_len(cv_nfolds))]),
            paired = TRUE
          )$p.value >= 0.05
        })
      ## NA's may result if CV prediction log scores are the same for the model
      ## specified by ind as for the model specified by best_params_ind (and so
      ## NA is guaranteed at at least the index of best_params_ind
      refit_params_inds[is.na(refit_params_inds)] <- TRUE 
      
      params <- cv_results[refit_params_inds, seq_len(ncol(cv_results) - cv_nfolds - 1)]
      ## only refit using the largest values of nrounds for each combination of
      ## other parameter values
      params_refit <- params %>%
        group_by_(.dots = colnames(params)[colnames(params) != "nrounds"]) %>%
        summarize(nrounds = max(nrounds)) %>%
        as.data.frame()
      fit <- list()
      for(single_params_ind in seq_len(nrow(params_refit))) {
        single_params <- params_refit[single_params_ind, , drop = FALSE]
        attr(single_params, "out.attrs") <- NULL
        single_params <- as.list(single_params)
        single_params <- c(single_params,
          base_params[!(names(base_params) %in% names(single_params))])
        
        fit[[single_params_ind]] <- xgb.train(
          params = single_params,
          data = dtrain,
          nrounds = single_params$nrounds,
          obj = obj_deriv_fn,
          verbose = 0
        ) %>%
          xgb.save.raw()
      }
    } else if(identical(cv_refit, "none")) {
      fit <- NULL
    } else {
      warning("Invalid option for cv_refit: must be one of 'best', 'ttest', or 'none'; treating as 'none'")
      fit <- NULL
    }
  } # end code for cross-validation for parameter selection
  
  ## return
  if(is.null(cv_params)) {
    return(structure(
      list(fit = fit,
        formula = formula,
        model_scores = model_scores,
        dtrain_Rmatrix = dtrain_Rmatrix,
        params = params,
        num_models = ncol(model_scores)),
      class = "xgbstack"
    ))
  } else {
    return(structure(
      list(fit = fit,
        formula = formula,
        model_scores = model_scores,
        dtrain_Rmatrix = dtrain_Rmatrix,
        params = params,
        params_refit = if(exists("params_refit")) {
            params_refit
          } else {
            NULL
          },
        num_models = ncol(model_scores),
        cv_folds = cv_folds,
        cv_results = cv_results),
      class = "xgbstack"
    ))
  }
}
