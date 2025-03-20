rm(list = ls())
set.seed(42)

# Load required libraries
library(data.table)
library(mlr3verse)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(knitr)
library(rpart)
library(rpart.plot)
library(GGally)
library(corrplot)
library(yardstick)

#------------------------------------------------------------------------------
# 1. Data loading and preliminary exploration
#------------------------------------------------------------------------------

# Import data
telecom_data <- read.csv("telecom.csv")

# Basic statistical summary
skim_result <- skimr::skim(telecom_data)
skim_result

# View the distribution of the target variable
table(telecom_data$Churn)
churn_rate <- mean(telecom_data$Churn == "Yes") * 100
cat("\nCustomer Churn Rate:", round(churn_rate, 2), "%\n")

# Visualize the churn distribution
churn_counts <- telecom_data %>%
  group_by(Churn) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

churn_plot <- ggplot(churn_counts, aes(x = Churn, y = count, fill = Churn)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")),
            position = position_stack(vjust = 0.5)) +
  labs(title = "Distribution of Customer Churn",
       x = "Churn",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "coral"))
churn_plot

# Make sure the SeniorCitizen variable is converted to factor type
telecom_data$SeniorCitizen <- as.factor(ifelse(telecom_data$SeniorCitizen == 1, "Yes", "No"))

# Make sure all character variables are converted to factors
telecom_data <- telecom_data %>%
  mutate_if(is.character, as.factor)

# Process Internet-related service variables and classify "No internet service" as "No"
internet_dependent_vars <- c("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                             "TechSupport", "StreamingTV", "StreamingMovies")

for(var in internet_dependent_vars) {
  telecom_data[[var]] <- factor(ifelse(telecom_data[[var]] == "No internet service",
                                       "No", as.character(telecom_data[[var]])))
}

telecom_data$MultipleLines <- factor(ifelse(telecom_data$MultipleLines == "No phone service",
                                            "No", as.character(telecom_data$MultipleLines)))

#------------------------------------------------------------------------------
# 2. Exploratory Data Analysis
#------------------------------------------------------------------------------

# View the distribution of numerical variables
telecom_data %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "steelblue", bins = 30) +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Numerical Variables")

# View the distribution of service duration by churn status
p1 <- ggplot(telecom_data, aes(x = Churn, y = tenure, fill = Churn)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Service duration distribution by churn status",
       x = "Churn status", y = "Service duration (month)") +
  scale_fill_brewer(palette = "Set1")
p1

# Check the relationship between contract type and churn rate
p2 <- telecom_data %>%
  ggplot(aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Relationship between contract type and churn rate",
       x = "Contract type", y = "Proportion") +
  scale_fill_brewer(palette = "Set1")
p2

# Check the relationship between Internet services and churn rate
p3 <- telecom_data %>%
  ggplot(aes(x = InternetService, fill = Churn)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "The relationship between Internet services and churn rate",
       x = "Internet services", y = "Proportion") +
  scale_fill_brewer(palette = "Set1")
p3

# Multivariate analysis of numerical variables using GGally
ggpairs_plot <- telecom_data %>%
  select(tenure, MonthlyCharges, TotalCharges, Churn) %>%
  ggpairs(aes(color = Churn))
ggpairs_plot

# Correlation plot of numerical variables
cor_data <- telecom_data %>%
  select(tenure, MonthlyCharges, TotalCharges) %>%
  cor(use = "com")

corrplot(cor_data, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7)

#------------------------------------------------------------------------------
# Model fitting and evaluation
#------------------------------------------------------------------------------

# Define the classification task and use "Yes" as the positive class for loss
telecom_task <- TaskClassif$new(id = "TelecomChurn",
                                backend = telecom_data,
                                target = "Churn",
                                positive = "Yes")

# Define 5-fold cross-validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(telecom_task)

# ----2.1 Basic model implementation ----

# Baseline classifier
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")

# Decision tree model
lrn_cart <- lrn("classif.rpart", predict_type = "prob")

# Fit and evaluate the basic model
set.seed(42)
res_basic <- benchmark(data.table(
  task = list(telecom_task),
  learner = list(lrn_baseline,
                 lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)

# Basic performance metrics
metrics_basic <- list(
  msr("classif.ce"), # Classification error
  msr("classif.acc"), # Accuracy
  msr("classif.auc"), # Area under the ROC curve
  msr("classif.precision"), # Precision
  msr("classif.recall"), # Recall
  msr("classif.fbeta", beta = 1) # F1 score
)

# Get basic model performance
performance_basic <- res_basic$aggregate(metrics_basic)

# Create performance table
performance_basic_table <- as.data.frame(performance_basic) %>%
  mutate(
    Model = c("Baseline", "CART"),
    Accuracy=1-classif.ce
  ) %>%
  select(Model, Accuracy, classif.auc, classif.precision, classif.recall, classif.fbeta) %>%
  rename(
    "AUC" = classif.auc,
    "Precision" = classif.precision,
    "Recall" = classif.recall,
    "F1 Score" = classif.fbeta
  )

kable(performance_basic_table, digits = 3)
# write.csv(performance_basic_table, "basic_model_performance.csv", row.names = FALSE)

# Visualize basic model performance
performance_basic_long <- performance_basic_table %>%
  select(Model, Accuracy, AUC, Precision, Recall, `F1 Score`) %>%
  pivot_longer(cols = -Model,
               names_to = "Metric", 
               values_to = "Value")

basic_model_plot <- ggplot(performance_basic_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Basic Model Performance Comparison", 
       x = "Model", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
basic_model_plot


#------------------------------------------------------------------------------
# 3. Model Improvement
#------------------------------------------------------------------------------

# ---- 3.1 Implementing Advanced Models ----

# Logistic Regression 
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean") %>>%
  po("encode")

lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg_simple <- pl_missing %>>% po(lrn_log_reg)


# Random Forest Model
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
pl_ranger <- pl_missing %>>% po(lrn_ranger)

# xgb
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
pl_xgboost <- pl_missing %>>% po(lrn_ranger)

# Improved logistic regression model
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>% po(lrn_log_reg)

# ---- 3.2 Super Learner Implementation ----

# Define Super Learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Create Super Learner Pipeline
set.seed(42)

spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("learner_cv", pl_xgboost),
      po("nop") # This passes through the original features adjusted for
      # missingness to the super learner
    ))
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

res_spr <- resample(telecom_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

# Save Super Learner Pipeline Graph
spr_lrn$plot()

# ---- 4 Evaluate all improved models ----
# Run all improved models
set.seed(42)
res_improved <- benchmark(data.table(
  task       = list(telecom_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    pl_ranger,
                    pl_xgboost,
                    pl_log_reg,
                    spr_lrn),
  resampling = list(cv5)
), store_models = TRUE)

# Get the complete performance metrics
metrics_full <- list(
  msr("classif.ce"), # Classification error
  msr("classif.acc"), # Accuracy
  msr("classif.auc"), # Area under the ROC curve
  msr("classif.precision"), # Precision
  msr("classif.recall"), # Recall
  msr("classif.fbeta", beta = 1), # F1 score
  msr("classif.specificity") # Specificity
)

performance_full <- res_improved$aggregate(metrics_full)

performance_table <- as.data.frame(performance_full) %>%
  mutate(
    Model = c("Baseline", "CART", "Random Forest", 
              "XGBoost","Logistic Regression", "Super Learner"),
    Accuracy = 1 - classif.ce
  ) %>%
  select(Model, Accuracy, classif.auc, classif.precision, classif.recall, 
         classif.fbeta, classif.specificity) %>%
  rename(
    "AUC" = classif.auc,
    "Precision" = classif.precision,
    "Recall" = classif.recall,
    "F1 Score" = classif.fbeta,
    "Specificity" = classif.specificity
  )

kable(performance_table, digits = 3)


