
# Install and load required packages
required_packages <- c(
  "tidyverse", "lubridate", "caret", "rpart", "rpart.plot",
  "randomForest", "xgboost", "pROC", "readr", "DALEX",
  "e1071", "ggplot2", "broom"
)
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
invisible(lapply(required_packages, library, character.only = TRUE))


library(tidyverse)
library(e1071)  # for skewness function



# Additionally install text-mining packages for clustering
if (!require(tm   )) install.packages("tm")
if (!require(SnowballC)) install.packages("SnowballC")

library(tm)
library(SnowballC)
library(tibble)

# Read data into a tibble
df <- read_csv("Downloads/Cleaned_dataset_A3/A3_Dataset_2025.csv")

# Remove duplicate rows
df <- df %>% distinct()

# Fix column types: parse dates, convert ID columns to numeric, and characters to factors
df <- df %>%
  mutate(
    Date            = lubridate::dmy(Date),
    RecommendationId = as.numeric(RecommendationId),
    RequestId       = as.numeric(RequestId),
    AdviserID       = as.numeric(AdviserID),
    LifeID          = as.numeric(LifeId)
  ) %>%
  mutate(across(where(is.character), as.factor))

# Handle missing values: factor NAs -> "Missing", numeric NAs -> median
df <- df %>%
  mutate(across(where(is.factor), ~ forcats::fct_explicit_na(., "Missing"))) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Encode target Underwriter_NEOS as a binary factor
df <- df %>%
  mutate(
    Underwriter_NEOS = ifelse(Underwriter == "NEOS Life", 1, 0),
    Underwriter_NEOS = factor(Underwriter_NEOS, levels = c(0, 1))
  )

# Feature engineering: CommissionGroup and OccupationGroup
df <- df %>%
  mutate(
    CommissionGroup = case_when(
      str_detect(CommissionStructure, regex("Upfront", ignore_case = TRUE)) ~ "Upfront",
      str_detect(CommissionStructure, regex("Hybrid",  ignore_case = TRUE)) ~ "Hybrid",
      str_detect(CommissionStructure, regex("Level",   ignore_case = TRUE)) ~ "Level",
      str_detect(CommissionStructure, regex("Trail",   ignore_case = TRUE)) ~ "Trail",
      str_detect(CommissionStructure, regex("Bundled", ignore_case = TRUE)) ~ "Bundled",
      TRUE                                                                  ~ "Other"
    ),
    OccupationGroup = case_when(
      str_detect(Occupation, regex("health|nurse|doctor|pharmac", ignore_case = TRUE)) ~ "Healthcare",
      str_detect(Occupation, regex("teach|educat",                   ignore_case = TRUE)) ~ "Education",
      str_detect(Occupation, regex("finance|admin|account",          ignore_case = TRUE)) ~ "Finance/Admin",
      str_detect(Occupation, regex("trade|skilled|electrician|plumb",ignore_case = TRUE)) ~ "Trade/Skilled",
      str_detect(Occupation, regex("it|tech|software|developer",     ignore_case = TRUE)) ~ "IT/Tech",
      str_detect(Occupation, regex("sales|marketing",                ignore_case = TRUE)) ~ "Sales/Marketing",
      str_detect(Occupation, regex("manager|exec|officer",           ignore_case = TRUE)) ~ "Management",
      str_detect(Occupation, regex("labour|mechanic|driver|construction", ignore_case = TRUE)) ~ "Manual/Construction",
      SelfEmployed == "Yes"                                                                        ~ "Self-Employed",
      TRUE                                                                                         ~ "Other"
    )
  )

# Cluster raw Package names into 10 clusters
unique_pkgs <- df %>% distinct(Package) %>% pull(Package)
corpus <- VCorpus(VectorSource(unique_pkgs)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace)
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
mat <- as.matrix(dtm)[, colSums(as.matrix(dtm)) > 0]
set.seed(123)
km <- kmeans(mat, centers = 10, nstart = 25)
df_clusters <- tibble(
  Package        = unique_pkgs,
  PackageCluster = factor(km$cluster)
)
df <- df %>% left_join(df_clusters, by = "Package")

# Drop raw/intermediary columns
df <- df %>% select(-c(
  RecommendationId, RequestId, AdviserID, LifeID,
  Underwriter, CommissionStructure, Package,
  Occupation, Alternative,
  Time, ExternalRef, BECoverAmount, SeverityCoverAmount,
  BE, Severity, LifeId
))

# Transform skewed numeric variables
skew_vars <- c(
  "Premium","AnnualisedPremium","InsideSuperPremium","OutsideSuperPremium",
  "AnnualIncome","LifeCoverAmount","TPDCoverAmount","TraumaCoverAmount",
  "IPCoverAmount","IndexationRate"
)
df <- df %>%
  mutate(across(all_of(skew_vars), log1p, .names = "{.col}_log")) %>%
  select(-all_of(skew_vars)) %>%
  mutate(
    across(where(is.numeric), ~ ifelse(is.infinite(.), NA_real_, .)),
    across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
  )

# Train/test split (70/30) stratified on Underwriter_NEOS
set.seed(123)
train_index     <- createDataPartition(df$Underwriter_NEOS, p = 0.7, list = FALSE)
train_data      <- df[train_index, ]
validation_data <- df[-train_index, ]

table(train_data$Underwriter_NEOS)
table(validation_data$Underwriter_NEOS)

# Align factor levels between train and validation
autos <- names(train_data)[sapply(train_data, is.factor)]
for (col in autos) {
  validation_data[[col]] <- factor(validation_data[[col]], levels = levels(train_data[[col]]))
}

# Remove any remaining rows with missing values
# 1) Identify which columns are factors in train_data
factor_cols <- names(train_data)[sapply(train_data, is.factor)]

# 2) Align factor levels in validation_data
for (col in factor_cols) {
  validation_data[[col]] <- factor(
    validation_data[[col]],
    levels = levels(train_data[[col]])
  )
}

# 3) Now drop any rows with remaining NAs (after alignment)
train_data      <- train_data      %>% filter(if_all(everything(), ~ !is.na(.)))
validation_data <- validation_data %>% filter(if_all(everything(), ~ !is.na(.)))
# ------------------------------
# Exploratory: Print example cluster members
# ------------------------------
df_clusters %>%
  group_by(PackageCluster) %>%
  slice_head(n = 8) %>%
  arrange(PackageCluster) %>%
  print(n = 80)


# ===============================
# Logistic regression
# ===============================
model_aug <- glm(
  Underwriter_NEOS ~ AgeNext + Gender + SmokerStatus + HomeState + SelfEmployed
  + CommissionGroup + OccupationGroup + PackageCluster,
  data   = train_data,
  family = binomial
)
summary(model_aug)
pred_aug     <- predict(model_aug, newdata = validation_data, type = "response")
pred_aug_lbl <- factor(ifelse(pred_aug > 0.5, 1, 0), levels = c(0,1))
conf_mat_aug <- confusionMatrix(pred_aug_lbl, validation_data$Underwriter_NEOS)
print(conf_mat_aug)
roc_aug      <- roc(validation_data$Underwriter_NEOS, pred_aug)
plot(roc_aug, main = "ROC Curve: Augmented Logistic Model")
cat("AUC (Augmented Model) =", auc(roc_aug), "\n")

# Load ML metrics library
library(MLmetrics)

# Create predicted label using default 0.5
pred_lbl_0.5 <- ifelse(pred_aug > 0.5, 1, 0)

# Calculate metrics
precision <- Precision(y_pred = pred_lbl_0.5, y_true = validation_data$Underwriter_NEOS)
recall    <- Recall(y_pred = pred_lbl_0.5, y_true = validation_data$Underwriter_NEOS)
f1        <- F1_Score(y_pred = pred_lbl_0.5, y_true = validation_data$Underwriter_NEOS)

cat("Precision (0.5 threshold):", round(precision, 4), "\n")
cat("Recall    (0.5 threshold):", round(recall, 4), "\n")
cat("F1 Score  (0.5 threshold):", round(f1, 4), "\n")

thresholds <- seq(0.1, 0.9, by = 0.05)
f1_scores <- sapply(thresholds, function(t) {
  pred_lbl <- ifelse(pred_aug > t, 1, 0)
  F1_Score(y_pred = pred_lbl, y_true = validation_data$Underwriter_NEOS)
})

# Plot F1 vs threshold
plot(thresholds, f1_scores, type = "b", pch = 19,
     xlab = "Threshold", ylab = "F1 Score", main = "F1 vs Threshold")

# Best threshold
best_thresh <- thresholds[which.max(f1_scores)]
cat("Best Threshold (by F1):", best_thresh, "\n")

# ------------------------------
# 4. Per‐Product Logistic Models
# ------------------------------
products <- c("Life","TPD","Trauma","IP")

# Convert Yes/No/Missing to 1/0/NA and ensure factor levels
for (prod in products) {
  train_data[[prod]] <- ifelse(train_data[[prod]] == "Yes", 1,
                               ifelse(train_data[[prod]] == "No", 0, NA))
  validation_data[[prod]] <- ifelse(validation_data[[prod]] == "Yes", 1,
                                    ifelse(validation_data[[prod]] == "No", 0, NA))
  
  train_data[[prod]] <- factor(train_data[[prod]], levels = c(0, 1))
  validation_data[[prod]] <- factor(validation_data[[prod]], levels = c(0, 1))
}

# Create empty list to store models
models <- list()

# Loop through each product
for (p in products) {
  cat("\n---", p, " Recommendation Model ---\n")
  
  # Filter out NA values
  train_sub <- train_data %>% filter(!is.na(.data[[p]]))
  valid_sub <- validation_data %>% filter(!is.na(.data[[p]]))
  
  # Fit logistic regression model
  f <- as.formula(paste0(
    p,
    " ~ AgeNext + Gender + SmokerStatus + HomeState + SelfEmployed +",
    " CommissionGroup + OccupationGroup + PackageCluster"
  ))
  models[[p]] <- glm(f, data = train_sub, family = binomial)
  print(summary(models[[p]]))
  
  # Predict probabilities
  pred_prob <- predict(models[[p]], newdata = valid_sub, type = "response")
  pred_lbl  <- factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0, 1))
  actual_lbl <- valid_sub[[p]]
  
  # Check for presence of both classes
  if (length(unique(actual_lbl)) < 2) {
    cat(p, ": only one class present in validation data — skipping confusionMatrix\n")
  } else {
    cm <- confusionMatrix(pred_lbl, actual_lbl)
    auc_val <- auc(roc(actual_lbl, pred_prob))
    
    cat(p, "— AUC:", round(auc_val, 4), "\n")
    cat("Confusion Matrix:\n")
    print(cm$table)
    cat("Accuracy:", cm$overall["Accuracy"], "\n")
    cat("Sensitivity:", cm$byClass["Sensitivity"], "\n")
    cat("Specificity:", cm$byClass["Specificity"], "\n")
  }
}



library(dplyr)
library(pROC)
library(MLmetrics)

products <- c("Life", "TPD", "Trauma", "IP")

for (p in products) {
  cat("\n======", p, "======\n")
  
  # Prepare data: remove NAs
  train_sub <- train_data %>% filter(!is.na(.data[[p]]))
  valid_sub <- validation_data %>% filter(!is.na(.data[[p]]))
  
  # Fit logistic model
  f <- as.formula(paste0(
    p, " ~ AgeNext + Gender + SmokerStatus + HomeState + SelfEmployed +",
    " CommissionGroup + OccupationGroup + PackageCluster"
  ))
  model <- glm(f, data = train_sub, family = binomial)
  
  # Predict probabilities
  prob <- predict(model, newdata = valid_sub, type = "response")
  actual <- valid_sub[[p]]
  
  # Define thresholds
  thresholds <- seq(0.1, 0.9, by = 0.01)
  
  # Initialize storage
  f1_list <- numeric(length(thresholds))
  youden_list <- numeric(length(thresholds))
  
  for (i in seq_along(thresholds)) {
    t <- thresholds[i]
    pred <- ifelse(prob > t, 1, 0)
    pred <- factor(pred, levels = c(0, 1))
    actual <- factor(actual, levels = c(0, 1))
    
    cm <- caret::confusionMatrix(pred, actual)
    sens <- cm$byClass["Sensitivity"]
    spec <- cm$byClass["Specificity"]
    
    f1_list[i] <- F1_Score(y_pred = pred, y_true = actual)
    youden_list[i] <- sens + spec - 1
  }
  
  # Find best thresholds
  best_thresh_f1 <- thresholds[which.max(f1_list)]
  best_thresh_youden <- thresholds[which.max(youden_list)]
  
  cat("Best Threshold (F1)     :", best_thresh_f1, "\n")
  cat("Best Threshold (Youden) :", best_thresh_youden, "\n")
  
  # Plot
  plot(thresholds, f1_list, type = "l", col = "blue", ylim = c(0, 1),
       xlab = "Threshold", ylab = "Score", main = paste("F1 & Youden for", p))
  lines(thresholds, youden_list, col = "red")
  abline(v = best_thresh_f1, col = "blue", lty = 2)
  abline(v = best_thresh_youden, col = "red", lty = 2)
  legend("bottomright", legend = c("F1 Score", "Youden Index"),
         col = c("blue", "red"), lty = 1)
}

# ================================
# Define best thresholds (F1)
# ================================
best_thresh_f1 <- list(
  Life   = 0.77,
  TPD    = 0.82,
  Trauma = 0.74,
  IP     = 0.71
)

# Load required metrics
library(MLmetrics)
library(pROC)

# Initialize result dataframe
perf_results <- data.frame(
  Product    = character(),
  Accuracy   = numeric(),
  Precision  = numeric(),
  Recall     = numeric(),
  F1_Score   = numeric(),
  AUC        = numeric(),
  Threshold  = numeric(),
  stringsAsFactors = FALSE
)

# Evaluate each product
for (p in products) {
  # Prepare data (filter out missing)
  valid_sub <- validation_data %>% filter(!is.na(.data[[p]]))
  actual    <- as.numeric(as.character(valid_sub[[p]]))
  
  # Predict probability
  pred_prob <- predict(models[[p]], newdata = valid_sub, type = "response")
  threshold <- best_thresh_f1[[p]]
  pred_lbl  <- ifelse(pred_prob > threshold, 1, 0)
  
  # Compute metrics
  acc  <- mean(pred_lbl == actual)
  prec <- Precision(y_pred = pred_lbl, y_true = actual)
  rec  <- Recall(y_pred = pred_lbl, y_true = actual)
  f1   <- F1_Score(y_pred = pred_lbl, y_true = actual)
  auc  <- auc(roc(actual, pred_prob))
  
  # Append to results table
  perf_results <- rbind(perf_results, data.frame(
    Product    = p,
    Accuracy   = round(acc, 4),
    Precision  = round(prec, 4),
    Recall     = round(rec, 4),
    F1_Score   = round(f1, 4),
    AUC        = round(auc, 4),
    Threshold  = threshold
  ))
}

# Display results
print(perf_results)

# Optional: Plot comparison
library(ggplot2)
library(tidyr)

perf_long <- perf_results %>%
  pivot_longer(cols = c("Accuracy", "Precision", "Recall", "F1_Score", "AUC"),
               names_to = "Metric", values_to = "Value")

ggplot(perf_long, aes(x = Product, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(title = "Model Performance by Product (F1 Threshold)", y = "Score") +
  theme_minimal()

# Improved color palette
ggplot(perf_long, aes(x = Product, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  scale_fill_manual(
    values = c(
      Accuracy  = "#999999",  # grey
      AUC       = "#E69F00",  # orange
      F1_Score  = "#56B4E9",  # light blue
      Precision = "#009E73",  # green
      Recall    = "#0072B2"   # dark blue
    )
  ) +
  labs(
    title = "Model Performance by Product (F1 Threshold)",
    y = "Score",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")


library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(pROC)
library(caret)
library(dplyr)
library(tibble)

# ===============================
# Clean Data (remove rows with any NAs)
# ===============================
train_data_clean <- train_data %>% drop_na()
validation_data_clean <- validation_data %>% drop_na()

# ===============================
# Helper for Metrics
# ===============================
calculate_metrics <- function(cm, name) {
  acc  <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  cat(sprintf("%s → Accuracy: %.4f | Sensitivity: %.4f | Specificity: %.4f\n", name, acc, sens, spec))
}

# ===============================
# Decision Tree (All Variables)
# ===============================
tree_model2 <- rpart(Underwriter_NEOS ~ ., data = train_data_clean, method = "class")
rpart.plot(tree_model2, extra = 101, box.palette = "RdBu", fallen.leaves = TRUE,
           main = "Decision Tree (All Variables)")
tree_pred2 <- predict(tree_model2, validation_data_clean, type = "class")
tree_prob2 <- predict(tree_model2, validation_data_clean, type = "prob")[,2]
tree_cm2   <- confusionMatrix(tree_pred2, validation_data_clean$Underwriter_NEOS)
tree_auc2  <- auc(roc(validation_data_clean$Underwriter_NEOS, tree_prob2))
cat("\nDecision Tree (All Variables) — AUC:", round(tree_auc2, 4), "\n")
calculate_metrics(tree_cm2, "Decision Tree")

# ===============================
# Random Forest (All Variables)
# ===============================
rf_model2 <- randomForest(Underwriter_NEOS ~ ., data = train_data_clean, ntree = 100, importance = TRUE)
rf_pred2  <- predict(rf_model2, validation_data_clean, type = "class")
rf_prob2  <- predict(rf_model2, validation_data_clean, type = "prob")[,2]
rf_cm2    <- confusionMatrix(rf_pred2, validation_data_clean$Underwriter_NEOS)
rf_auc2   <- auc(roc(validation_data_clean$Underwriter_NEOS, rf_prob2))
cat("\nRandom Forest — AUC:", round(rf_auc2, 4), "\n")
calculate_metrics(rf_cm2, "Random Forest")
# Top 10 Random Forest features
rf_imp2 <- importance(rf_model2)
top_rf2 <- rownames(rf_imp2)[order(rf_imp2[,"MeanDecreaseGini"], decreasing = TRUE)][1:10]
cat("\nTop 10 RF features:\n"); print(top_rf2)

# ===============================
# XGBoost (All Variables)
# ===============================
encode_mat <- function(df){
  df %>%
    mutate(across(everything(), ~ as.numeric(as.factor(.)))) %>%
    as.matrix()
}

dtrain2 <- xgb.DMatrix(data = encode_mat(train_data_clean %>% select(-Underwriter_NEOS)),
                       label = as.numeric(train_data_clean$Underwriter_NEOS) - 1)
dval2 <- xgb.DMatrix(data = encode_mat(validation_data_clean %>% select(-Underwriter_NEOS)),
                     label = as.numeric(validation_data_clean$Underwriter_NEOS) - 1)

params2 <- list(objective = "binary:logistic", eval_metric = "auc")
xgb_mod2 <- xgb.train(params2, dtrain2, nrounds = 100, verbose = 0)
xgb_prob2 <- predict(xgb_mod2, dval2)
xgb_pred2 <- factor(ifelse(xgb_prob2 > 0.5, 1, 0), levels = c(0, 1))
y_val2 <- factor(as.numeric(validation_data_clean$Underwriter_NEOS) - 1, levels = c(0, 1))
xgb_cm2 <- confusionMatrix(xgb_pred2, y_val2)
xgb_auc2 <- auc(roc(validation_data_clean$Underwriter_NEOS, xgb_prob2))
cat("\nXGBoost — AUC:", round(xgb_auc2, 4), "\n")
calculate_metrics(xgb_cm2, "XGBoost")

# ===============================
# Model Comparison Summary Table
# ===============================
tb_perf <- tibble(
  Model       = c("Decision Tree", "Random Forest", "XGBoost"),
  AUC         = c(tree_auc2, rf_auc2, xgb_auc2),
  Accuracy    = c(
    mean(tree_pred2 == validation_data_clean$Underwriter_NEOS),
    mean(rf_pred2   == validation_data_clean$Underwriter_NEOS),
    mean(xgb_pred2  == y_val2)
  ),
  Sensitivity = c(
    tree_cm2$byClass["Sensitivity"],
    rf_cm2$byClass["Sensitivity"],
    xgb_cm2$byClass["Sensitivity"]
  ),
  Specificity = c(
    tree_cm2$byClass["Specificity"],
    rf_cm2$byClass["Specificity"],
    xgb_cm2$byClass["Specificity"]
  )
) %>% mutate(across(where(is.numeric), round, 4))

print(tb_perf)

# ===============================
# Plot ROC Curves
# ===============================
roc_tree2 <- roc(validation_data_clean$Underwriter_NEOS, tree_prob2)
roc_rf2   <- roc(validation_data_clean$Underwriter_NEOS, rf_prob2)
roc_xgb2  <- roc(validation_data_clean$Underwriter_NEOS, xgb_prob2)

plot(roc_tree2, col = "blue", main = "ROC Curves")
lines(roc_rf2,  col = "darkgreen")
lines(roc_xgb2, col = "red")
legend("bottomright",
       legend = c("Decision Tree", "Random Forest", "XGBoost"),
       col    = c("blue", "darkgreen", "red"),
       lwd    = 2)

# ===============================
# DALEX Interpretation - Random Forest Model
# ===============================

# Convert factor target to numeric: 0/1
y_numeric <- as.numeric(as.character(train_data$Underwriter_NEOS))

# Create explainer
explainer_rf <- explain(
  model = rf_model2,
  data = train_data %>% select(-Underwriter_NEOS),
  y = y_numeric,
  label = "Random Forest"
)

# 1. Feature Importance
rf_parts <- model_parts(explainer_rf)
plot(rf_parts) +
  ggtitle("DALEX - Feature Importance (Random Forest)") +
  theme_minimal()

# 2. Safe Top 5 Feature Selection
top5_features <- rf_parts %>%
  filter(variable != "_full_model_", variable %in% colnames(train_data)) %>%
  arrange(desc(dropout_loss)) %>%
  slice_head(n = 5) %>%
  pull(variable)

# 3. Plot PDPs
for (feat in top5_features) {
  pdp <- model_profile(explainer_rf, variables = feat)
  print(plot(pdp) +
          ggtitle(paste("Partial Dependence Plot -", feat)) +
          theme_minimal())
}

# ===============================
# DALEX PDP for OccupationGroup (Single Variable)
# ===============================

# Generate Partial Dependence Plot for OccupationGroup
pdp_occupation <- model_profile(explainer_rf, variables = "OccupationGroup")

# Plot it
plot(pdp_occupation) +
  ggtitle("Partial Dependence Plot - OccupationGroup (Random Forest)") +
  theme_minimal()

# ===============================
# DALEX PDP for PackageCluster (Single Variable)
# ===============================

# Generate PDP
pdp_package <- model_profile(explainer_rf, variables = "PackageCluster")

# Plot PDP
plot(pdp_package) +
  ggtitle("Partial Dependence Plot - PackageCluster (Random Forest)") +
  theme_minimal()

table(train_data$PackageCluster, train_data$Underwriter_NEOS)
df_clusters %>% filter(PackageCluster == 3) %>% slice_head(n = 10)
# ===============================
# ===============================
# Combine packages and cluster into a data frame
pkg_cluster_df <- df_clusters %>%
  mutate(Package = tolower(Package)) %>%
  filter(!is.na(Package))

# Create a corpus by cluster
pkg_terms_by_cluster <- pkg_cluster_df %>%
  group_by(PackageCluster) %>%
  summarise(text = paste(Package, collapse = " ")) %>%
  ungroup()

# Tokenize and calculate TF-IDF
library(tidytext)
tidy_clusters <- pkg_terms_by_cluster %>%
  unnest_tokens(word, text) %>%
  anti_join(get_stopwords()) %>%
  count(PackageCluster, word, sort = TRUE) %>%
  bind_tf_idf(word, PackageCluster, n)

# Show top 10 terms per cluster
top_terms <- tidy_clusters %>%
  group_by(PackageCluster) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup()

# Optional: View or print
print(top_terms, n = 100)

library(ggplot2)

top_terms %>%
  mutate(word = reorder_within(word, tf_idf, PackageCluster)) %>%
  ggplot(aes(x = word, y = tf_idf)) +
  geom_col(fill = "steelblue") +
  scale_x_reordered() +
  facet_wrap(~ PackageCluster, scales = "free") +
  coord_flip() +
  labs(title = "Top TF-IDF Terms per Package Cluster",
       x = "Term", y = "TF-IDF") +
  theme_minimal()

df_clusters %>%
  group_by(PackageCluster) %>%
  slice_sample(n = 5) %>%
  arrange(PackageCluster) %>%
  print(n = 50)

library(stopwords)



# ─────────────────────────────────────────────────────────────
#  Product Recommendation Volume Tracking by Quarter
# ─────────────────────────────────────────────────────────────

# Load libraries
library(tidyverse)
library(lubridate)
library(tidyr)
library(ggplot2)


# Filter for NEOS Life (where Underwriter_NEOS == 1)
df_neos <- df %>%
  filter(Underwriter_NEOS == 1)

# Convert product columns to binary 1/0
df_neos <- df_neos %>%
  mutate(
    Life   = ifelse(Life == "Yes", 1, 0),
    TPD    = ifelse(TPD == "Yes", 1, 0),
    Trauma = ifelse(Trauma == "Yes", 1, 0),
    IP     = ifelse(IP == "Yes", 1, 0)
  )

# Create Year, Quarter, QuarterIndex
df_neos <- df_neos %>%
  mutate(
    Year         = year(Date),
    QuarterNum   = quarter(Date),
    QuarterIndex = (Year - min(Year, na.rm = TRUE)) * 4 + QuarterNum,
    YearQuarter  = paste0(Year, " Q", QuarterNum)
  )

# Convert product fields to binary if needed (uncomment if using "Yes"/"No")
# df_neos <- df_neos %>%
#   mutate(
#     Life   = ifelse(Life == "Yes", 1, 0),
#     TPD    = ifelse(TPD == "Yes", 1, 0),
#     Trauma = ifelse(Trauma == "Yes", 1, 0),
#     IP     = ifelse(IP == "Yes", 1, 0)
#   )

# Aggregate recommendations by quarter
quarterly_counts_neos <- df_neos %>%
  group_by(YearQuarter) %>%
  summarise(
    Life_Count   = sum(Life, na.rm = TRUE),
    TPD_Count    = sum(TPD, na.rm = TRUE),
    Trauma_Count = sum(Trauma, na.rm = TRUE),
    IP_Count     = sum(IP, na.rm = TRUE),
    .groups = "drop"
  )

# Transform to long format for plotting
long_counts_neos <- quarterly_counts_neos %>%
  pivot_longer(cols = ends_with("_Count"),
               names_to = "Product", values_to = "Count")

# Plot with YearQuarter on x-axis
ggplot(long_counts_neos, aes(x = YearQuarter, y = Count, color = Product, group = Product)) +
  geom_line(linewidth = 1.2) +
  labs(
    title = "NEOS Product Recommendations Over Time",
    x = "Quarter",
    y = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# =========================
# 1. Load Required Libraries
# =========================


# =========================
# 3. Define Columns to Remove
# =========================
non_cluster_cols <- c("Date", "Underwriter", "Package", "CommissionStructure",
                      "Occupation", "Alternative", "ExternalRef", "Time",
                      "Life", "TPD", "Trauma", "IP", "Underwriter_NEOS")

# =========================
# 4. Create Filtered Copy for Clustering
# =========================
# Only keep numeric / factor variables suitable for clustering
df_clust <- df %>%
  select(-any_of(non_cluster_cols)) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

# Drop rows with NA in clustering data
na_rows <- complete.cases(df_clust)
df_clust <- df_clust[na_rows, ]
df_main <- df[na_rows, ]  # Keep only rows used in clustering

# =========================
# 5. Normalize and Reduce Dimensions
# =========================
df_scaled <- scale(df_clust)
pca_res <- prcomp(df_scaled, center = TRUE, scale. = TRUE)
df_pca <- as.data.frame(pca_res$x[, 1:5])  # Use top 5 PCs

# =========================
# 6. KMeans Clustering
# =========================
set.seed(123)
k_optimal <- 3
km_res <- kmeans(df_pca, centers = k_optimal, nstart = 25)

# =========================
# 7. Add Cluster Labels Back to Main Data
# =========================
df_main <- df_main %>%
  mutate(Cluster = factor(km_res$cluster))

# =========================
# 8. Summarise Segment Characteristics
# =========================
summary_table <- df_main %>%
  group_by(Cluster) %>%
  summarise(
    n_customers       = n(),
    avg_age           = mean(AgeNext, na.rm = TRUE),
    pct_smokers       = mean(SmokerStatus == "Smoker", na.rm = TRUE),
    pct_self_employed = mean(SelfEmployed == "Yes", na.rm = TRUE),
    avg_income        = mean(AnnualIncome_log, na.rm = TRUE),
    avg_premium       = mean(Premium_log, na.rm = TRUE),
    pct_NEOS          = mean(Underwriter_NEOS == 1, na.rm = TRUE)
  )

# =========================
# 9. Display Result Table
# =========================
kable(summary_table, digits = 2, caption = "Customer Segment Summary")

# --- Step 1: Define columns to use ONLY for clustering
clustering_cols <- df %>%
  select(-any_of(c("Date", "Underwriter", "Package", "CommissionStructure",
                   "Occupation", "Alternative", "ExternalRef", "Time",
                   "Life", "TPD", "Trauma", "IP", "Underwriter_NEOS"))) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

# --- Step 2: Find complete rows in clustering subset
valid_rows <- complete.cases(clustering_cols)

# --- Step 3: Cleaned versions of clustering and full dataset
df_clust <- clustering_cols[valid_rows, ]
df_clean <- df[valid_rows, ]  # ✅ Retains all original variables like Premium_log

# --- Step 4: Scale and apply PCA
df_scaled <- scale(df_clust)
pca_res <- prcomp(df_scaled)
df_pca <- as.data.frame(pca_res$x[, 1:5])

wss <- map_dbl(1:10, ~ kmeans(df_pca, centers = .x, nstart = 10)$tot.withinss)
plot(1:10, wss, type = "b", main = "Elbow Plot")
# --- Step 5: KMeans clustering
set.seed(123)
km_res <- kmeans(df_pca, centers = 3, nstart = 25)

# --- Step 6: Attach cluster label to full df
df_clean$Cluster <- factor(km_res$cluster)

# --- Step 7: NEOS Profiling (Updated to use Premium_log)
library(dplyr)
neos_summary <- df_clean %>%
  group_by(Cluster, Underwriter_NEOS) %>%
  summarise(
    count = n(),
    avg_age = mean(AgeNext, na.rm = TRUE),
    pct_smokers = mean(SmokerStatus == "Smoker", na.rm = TRUE),
    pct_self_employed = mean(SelfEmployed == "Yes", na.rm = TRUE),
    avg_income = mean(AnnualIncome_log, na.rm = TRUE),
    avg_premium = mean(Premium_log, na.rm = TRUE),  # ✅ Use log-transformed premium
    .groups = "drop"
  )

# --- Step 8: Display table
library(knitr)
kable(neos_summary, digits = 2, caption = "NEOS vs Non-NEOS Profile by Cluster")

# --- Step 1: Convert to long format for plotting
library(tidyr)
neos_long <- neos_summary %>%
  pivot_longer(cols = c(avg_age, pct_smokers, pct_self_employed, avg_income, avg_premium),
               names_to = "Metric", values_to = "Value") %>%
  mutate(
    Underwriter_NEOS = factor(Underwriter_NEOS, labels = c("Non-NEOS", "NEOS"))
  )

# --- Step 2: Plot bar charts

library(ggplot2)
ggplot(neos_long, aes(x = Cluster, y = Value, fill = Underwriter_NEOS)) +
  geom_col(position = "dodge") +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(
    title = "NEOS vs Non-NEOS Profiles by Customer Segment (Cluster)",
    x = "Customer Segment (Cluster)",
    y = "Value",
    fill = "Underwriter"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 12),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )
###################
#NEOS vs others per product type with random forest
#########################


products <- c("Life", "TPD", "Trauma", "IP")
rf_results <- list()

for (prod in products) {
  cat("\n==============================\n")
  cat("Random Forest - NEOS vs Others for", prod, "\n")
  
  # 1. Filter rows where the product is recommended ("Yes")
  df_prod <- df %>%
    filter(.data[[prod]] == "Yes")
  
  # 2. Train/test split (70/30) stratified on Underwriter_NEOS
  set.seed(123)
  idx <- createDataPartition(df_prod$Underwriter_NEOS, p = 0.7, list = FALSE)
  train_p <- df_prod[idx, ]
  valid_p <- df_prod[-idx, ]
  
  # 3. Fit random forest
  rf_model <- randomForest(
    Underwriter_NEOS ~ .,
    data = train_p %>% select(-all_of(products), -Date),  # drop other product cols
    ntree = 100,
    importance = TRUE
  )
  
  # 4. Predict
  pred_class <- predict(rf_model, valid_p, type = "class")
  pred_prob  <- predict(rf_model, valid_p, type = "prob")[, 2]
  actual     <- valid_p$Underwriter_NEOS
  
  # 5. Evaluate
  cm <- confusionMatrix(pred_class, actual)
  auc_val <- auc(actual, pred_prob)
  
  cat("AUC:", round(auc_val, 4), "\n")
  print(cm$table)
  cat("Accuracy:", round(cm$overall["Accuracy"], 4), "\n")
  cat("Sensitivity:", round(cm$byClass["Sensitivity"], 4), "\n")
  cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
  
  # 6. Feature importance
  imp_df <- importance(rf_model)
  imp_df <- data.frame(Feature = rownames(imp_df), MeanDecreaseGini = imp_df[, "MeanDecreaseGini"])
  imp_df <- imp_df %>% arrange(desc(MeanDecreaseGini))
  top10 <- imp_df %>% slice_head(n = 10)
  
  print(top10)
  
  # 7. Optional: Plot feature importance
  p <- ggplot(top10, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = paste("Top Features for", prod),
         x = "Feature", y = "Mean Decrease in Gini") +
    theme_minimal()
  
  print(p)
  
  # 8. Store results
  rf_results[[prod]] <- list(model = rf_model, auc = auc_val, cm = cm, importance = imp_df)
}

# ------------------------
# DALEX Feature Importance + PDPs
# ------------------------

# 1. Prepare explainer (convert target to numeric: 0 = Other, 1 = NEOS)
explainer <- explain(
  model = rf_model,
  data  = train_p %>% select(-all_of(products), -Date, -Underwriter_NEOS),
  y     = as.numeric(train_p$Underwriter_NEOS) - 1,
  label = paste("RF -", prod)
)

# 2. Feature importance (exclude _baseline_ and _full_model_)
imp <- model_parts(explainer)

top5 <- imp %>%
  filter(!variable %in% c("_full_model_", "_baseline_")) %>%
  arrange(desc(dropout_loss)) %>%
  slice_head(n = 5)

# Print top 5 variables
cat("\nTop 5 DALEX Variables for", prod, ":\n")
print(top5)

# 3. PDPs for top 5
for (v in top5$variable) {
  pdp <- model_profile(explainer, variables = v)
  print(
    plot(pdp) +
      ggtitle(paste("Partial Dependence Plot -", v, "(", prod, ")")) +
      theme_minimal()
  )
}



# Check distribution
table(df$PackageCluster, df$Underwriter_NEOS)