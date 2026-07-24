suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(pbkrtest)
})


duckdb_path <- "data/trials.duckdb"   # CHANGE
table_ref <- "trials.analytics_gold.mart_overall_analysis"

debug_year_min <- NULL
debug_year_max <- NULL

pb_nsim <- 200

con <- dbConnect(duckdb::duckdb(), duckdb_path, read_only = TRUE)

sql <- paste0("
  SELECT
    name1,
    location,
    year,
    experiment_name,
    plot,
    env_type,
    o_a_score
  FROM ", table_ref, "
  WHERE o_a_score IS NOT NULL
")

df <- dbGetQuery(con, sql)

dbDisconnect(con, shutdown = TRUE)

if (nrow(df) == 0) {
  stop("No rows returned. Check duckdb_path and table_ref.")
}

df$name1 <- as.factor(df$name1)
df$location <- as.factor(df$location)
df$year <- as.integer(df$year)
df$experiment_name <- as.factor(df$experiment_name)
df$plot <- suppressWarnings(as.integer(df$plot))
df$o_a_score <- as.numeric(df$o_a_score)

if (!is.null(debug_year_min)) df <- df[df$year >= debug_year_min, ]
if (!is.null(debug_year_max)) df <- df[df$year <= debug_year_max, ]

cat("\n--- Dataset summary ---\n")
cat("Rows:", nrow(df), "\n")
cat("Distinct varieties (name1):", nlevels(df$name1), "\n")
cat("Distinct locations:", nlevels(df$location), "\n")
cat("Year range:", min(df$year, na.rm=TRUE), "-", max(df$year, na.rm=TRUE), "\n")


# E1: location × year
df$env_id_loc_year <- interaction(df$location, df$year, drop = TRUE)

# E2: location × year × experiment_name
df$env_id_loc_year_exp <- interaction(df$location, df$year, df$experiment_name, drop = TRUE)

cat("\n--- Environment counts ---\n")
cat("Distinct env_id (location×year):", nlevels(df$env_id_loc_year), "\n")
cat("Distinct env_id (location×year×experiment):", nlevels(df$env_id_loc_year_exp), "\n")

var_table <- function(model) {
  vc <- as.data.frame(VarCorr(model))
  out <- data.frame(
    component = vc$grp,
    variance  = vc$vcov
  )

  if (!any(out$component == "Residual")) {
    resid_var <- attr(VarCorr(model), "sc")^2
    out <- rbind(out, data.frame(component = "Residual", variance = resid_var))
  }

  out$pct <- 100 * out$variance / sum(out$variance)
  out <- out[order(out$pct, decreasing = TRUE), ]
  rownames(out) <- NULL
  out
}


fit_m4_reml <- function(env_col) {
  form <- as.formula(paste0(
    "o_a_score ~ 1 + (1|name1) + (1|", env_col, ") + (1|name1:", env_col, ")"
  ))

  lme4::lmer(
    formula = form,
    data = df,
    REML = TRUE,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )
}

cat("\n--- Fitting REML variance decomposition models ---\n")

m4_e1_reml <- fit_m4_reml("env_id_loc_year")
m4_e2_reml <- fit_m4_reml("env_id_loc_year_exp")

vt_e1 <- var_table(m4_e1_reml)
vt_e2 <- var_table(m4_e2_reml)

cat("\n=== Variance components: E1 (location×year) ===\n")
print(vt_e1)

cat("\n=== Variance components: E2 (location×year×experiment) ===\n")
print(vt_e2)


fit_m3_ml <- function(env_col) {
  form <- as.formula(paste0(
    "o_a_score ~ 1 + (1|name1) + (1|", env_col, ")"
  ))
  lme4::lmer(
    formula = form,
    data = df,
    REML = FALSE,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )
}

fit_m4_ml <- function(env_col) {
  form <- as.formula(paste0(
    "o_a_score ~ 1 + (1|name1) + (1|", env_col, ") + (1|name1:", env_col, ")"
  ))
  lme4::lmer(
    formula = form,
    data = df,
    REML = FALSE,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )
}

cat("\n--- Fitting ML models for model comparison (M3 vs M4) ---\n")

m3_e1_ml <- fit_m3_ml("env_id_loc_year")
m4_e1_ml <- fit_m4_ml("env_id_loc_year")

m3_e2_ml <- fit_m3_ml("env_id_loc_year_exp")
m4_e2_ml <- fit_m4_ml("env_id_loc_year_exp")

cat("\n=== Likelihood ratio test (LRT): E1 (M3 vs M4) ===\n")
lrt_e1 <- anova(m3_e1_ml, m4_e1_ml)
print(lrt_e1)

cat("\n=== Likelihood ratio test (LRT): E2 (M3 vs M4) ===\n")
lrt_e2 <- anova(m3_e2_ml, m4_e2_ml)
print(lrt_e2)


cat("\n--- Parametric bootstrap model comparison (optional) ---\n")
cat("Running PBmodcomp with nsim =", pb_nsim, "for E1...\n")
pb_e1 <- pbkrtest::PBmodcomp(m4_e1_ml, m3_e1_ml, nsim = pb_nsim)
print(pb_e1)

cat("\nRunning PBmodcomp with nsim =", pb_nsim, "for E2...\n")
pb_e2 <- pbkrtest::PBmodcomp(m4_e2_ml, m3_e2_ml, nsim = pb_nsim)
print(pb_e2)


df$env_type <- as.factor(df$env_type)

cat("\n--- Fixed effect test: env_type (MED vs NE) ---\n")
m_fixed <- lmer(
  o_a_score ~ env_type + (1|name1) + (1|env_id_loc_year),
  data = df,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)
cat("\nFixed effect ANOVA (env_type):\n")
print(anova(m_fixed))


cat("\n--- Diagnostics: counts per group (summary) ---\n")
cat("\nObservations per variety (name1):\n")
print(summary(table(df$name1)))

cat("\nObservations per env_id_loc_year:\n")
print(summary(table(df$env_id_loc_year)))

cat("\nDone.\n")