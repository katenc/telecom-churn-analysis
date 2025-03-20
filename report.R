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
# 1. 数据加载与初步探索
#------------------------------------------------------------------------------

# 导入数据
telecom_data <- read.csv("telecom.csv")

# 基本统计摘要
skim_result <- skimr::skim(telecom_data)
skim_result

# 查看目标变量的分布
table(telecom_data$Churn)
churn_rate <- mean(telecom_data$Churn == "Yes") * 100
cat("\nCustomer Churn Rate:", round(churn_rate, 2), "%\n")

# 可视化流失分布
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

# 确保SeniorCitizen变量被转换为因子类型
telecom_data$SeniorCitizen <- as.factor(ifelse(telecom_data$SeniorCitizen == 1, "Yes", "No"))

# 确保所有字符变量被转换为因子
telecom_data <- telecom_data %>%
  mutate_if(is.character, as.factor)

# 处理互联网相关服务变量，将"No internet service"归类为"No"
internet_dependent_vars <- c("OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                             "TechSupport", "StreamingTV", "StreamingMovies")

for(var in internet_dependent_vars) {
  telecom_data[[var]] <- factor(ifelse(telecom_data[[var]] == "No internet service", 
                                       "No", as.character(telecom_data[[var]])))
}

telecom_data$MultipleLines <- factor(ifelse(telecom_data$MultipleLines == "No phone service", 
                                            "No", as.character(telecom_data$MultipleLines)))


#------------------------------------------------------------------------------
# 2. 探索性数据分析
#------------------------------------------------------------------------------

# 查看数值变量分布
telecom_data %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "steelblue", bins = 30) +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Numerical Variables")

# 按流失状态查看服务时长分布
p1 <- ggplot(telecom_data, aes(x = Churn, y = tenure, fill = Churn)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Service duration distribution by churn status",
       x = "Churn status", y = "Service duration (month)") +
  scale_fill_brewer(palette = "Set1")
p1

# 检查合同类型与流失率的关系
p2 <- telecom_data %>%
  ggplot(aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Relationship between contract type and churn rate",
       x = "Contract type", y = "Proportion") +
  scale_fill_brewer(palette = "Set1")
p2

# 检查互联网服务与流失率的关系
p3 <- telecom_data %>%
  ggplot(aes(x = InternetService, fill = Churn)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "The relationship between Internet services and churn rate",
       x = "Internet services", y = "Proportion") +
  scale_fill_brewer(palette = "Set1")
p3

# 使用GGally进行数值变量的多变量分析
ggpairs_plot <- telecom_data %>%
  select(tenure, MonthlyCharges, TotalCharges, Churn) %>%
  ggpairs(aes(color = Churn))
ggpairs_plot

# 数值变量相关性图
cor_data <- telecom_data %>%
  select(tenure, MonthlyCharges, TotalCharges) %>%
  cor(use = "com")

corrplot(cor_data, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.7)


#------------------------------------------------------------------------------
# 2. 模型拟合与评估
#------------------------------------------------------------------------------

# 定义分类任务，将"Yes"作为流失的阳性类
telecom_task <- TaskClassif$new(id = "TelecomChurn",
                                backend = telecom_data,
                                target = "Churn",
                                positive = "Yes")

# 定义5折交叉验证重采样策略
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(telecom_task)

# ----2 .1 基本模型实现 ----

# 基线分类器
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")

# 决策树模型
lrn_cart <- lrn("classif.rpart", predict_type = "prob")


# 拟合和评估基本模型
set.seed(42)
res_basic <- benchmark(data.table(
  task       = list(telecom_task),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)

# 基本性能指标
metrics_basic <- list(
  msr("classif.ce"),        # 分类错误
  msr("classif.acc"),       # 准确率
  msr("classif.auc"),       # ROC曲线下面积
  msr("classif.precision"), # 精确率
  msr("classif.recall"),   # 召回率
  msr("classif.fbeta", beta = 1) # F1分数
)

# 获取基本模型性能
performance_basic <- res_basic$aggregate(metrics_basic)

# 创建性能表格
performance_basic_table <- as.data.frame(performance_basic) %>%
  mutate(
    Model = c("Baseline", "CART"),
    Accuracy = 1 - classif.ce
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

# 可视化基本模型性能
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
# 3. 模型改进
#------------------------------------------------------------------------------

# ---- 3.1 实现高级模型 ----

# 逻辑回归（需要编码分类变量）
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean") %>>%
  po("encode")

lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg_simple <- pl_missing %>>% po(lrn_log_reg)


# 随机森林模型
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
pl_ranger <- pl_missing %>>% po(lrn_ranger)

# xgb
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
pl_xgboost <- pl_missing %>>% po(lrn_ranger)

# 改进的逻辑回归模型
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>% po(lrn_log_reg)

# ---- 3.2 超级学习器实现 ----

# 定义超级学习器
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# 创建超级学习器管道
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

# 保存超级学习器管道图
spr_lrn$plot()

# ---- 4 评估所有改进模型 ----
# 运行所有改进模型
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

# 获取完整的性能指标
metrics_full <- list(
  msr("classif.ce"),        # 分类错误
  msr("classif.acc"),       # 准确率
  msr("classif.auc"),       # ROC曲线下面积
  msr("classif.precision"), # 精确率
  msr("classif.recall"),    # 召回率
  msr("classif.fbeta", beta = 1), # F1分数
  msr("classif.specificity") # 特异度
)

performance_full <- res_improved$aggregate(metrics_full)

# 创建性能表格
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


