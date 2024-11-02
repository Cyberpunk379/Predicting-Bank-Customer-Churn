#import packages
#install.packages("dplyr")
library(dplyr)
#install.packages("ggplot2")
library(ggplot2) 
#install.packages("caret")    # package for calculating sensitivity and specificity
library(caret)
#install.packages("caTools")    # Load the necessary package for sampling
library(caTools)
#install.packages("Amelia")   #missmap for missing value visual
library(Amelia)


#import and read file
getwd()
setwd("/Users/badboihy/Downloads/Visualizing & Analyzing Data with R - Methods & Tools/projects/Predicting Bank Customer Churn- A Machine Learning Approach")
getwd()
df.train <- read.csv('BankChurnDataset-2.csv')
head(df.train)
str(df.train)
summary(df.train)


summary(df.train$EstimatedSalary)

# Handling missing Estimated Salary missing Estimated Salary values 
df.train$EstimatedSalary[is.na(df.train$EstimatedSalary)] <- 
  median(df.train$EstimatedSalary, na.rm = TRUE)

# Checking out Estimated Salary
summary(df.train$EstimatedSalary)

# Age Imputation, handling missing values in age

impute_age <- function(age,class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- 42
        
      }else if (class[i] == 2){
        out[i] <- 37
        
      }else{
        out[i] <- 32
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}  

fixed.ages <- impute_age(df.train$Age, df.train$HasCrCard)
df.train$Age <- fixed.ages

summary(df.train$Age)

#Exploratory data analysis, finding out missing value

missmap(df.train, main="Bank Churn Data - Missings Map", 
        col=c("yellow", "black"), legend=TRUE)

# Remove ineffective features
options(repos = c(CRAN = "https://cran.rstudio.com"))


df.train <- select(df.train, -id, -CustomerId, -Surname)

# checking remaining columns 
head(df.train,3)
str(df.train)

#Converting features to factors
df.train$Geography      <- as.factor(df.train$Geography)
df.train$Gender         <- as.factor(df.train$Gender)
df.train$HasCrCard.     <- as.factor(df.train$HasCrCard)
df.train$IsActiveMember <- as.factor(df.train$IsActiveMember)

str(df.train)

df.train <- select(df.train, -HasCrCard.)

str(df.train)

# Exploratory data analysis using GGPlot

# For Age and Churn
ggplot(df.train, aes(x = Age, fill = factor(Exited))) + 
  geom_histogram(aes(y = ..density..), position = "identity", 
                 bins = 20, alpha = 0.5) +
  geom_density(alpha = 0.7) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Age Distribution by Churn", x = "Age", y = "Density") +
  theme_classic()


# For Balance and Churn
ggplot(df.train, aes(x = Balance, fill = factor(Exited))) + 
  geom_histogram(position = "identity", bins = 20, alpha = 0.5) +
  facet_grid(. ~ Exited) +
  scale_fill_manual(values = c("0" = "#DECBE4", "1" = "#B2182B")) +
  labs(title = "Balance Distribution by Churn", x = "Balance", y = "Count") +
  theme_dark()


# For NumOfProducts and Churn
ggplot(df.train, aes(x = factor(NumOfProducts), fill = factor(Exited))) + 
  geom_bar(position = "fill") +
  facet_wrap(~NumOfProducts) +
  scale_fill_manual(values = c("0" = "#1f77b4", "1" = "#ff7f0e")) +
  labs(title = "Churn by Number of Products", 
       x = "Number of Products", y = "Proportion") +
  theme_minimal()



# For IsActiveMember and Churn
ggplot(df.train, aes(x = factor(IsActiveMember), fill = factor(Exited))) + 
  geom_bar(position = "fill", alpha = 0.5) +
  scale_fill_manual(values = c("0" = "#619CFF", "1" = "#F564E3")) +
  labs(title = "Churn by Active Membership", 
       x = "Is Active Member", y = "Proportion") +
  theme_bw()


# For Geography and Churn
ggplot(df.train, aes(x = Geography, fill = factor(Exited))) + 
  geom_bar(position = "fill", alpha = 0.5) +
  scale_fill_brewer(palette = "Set1", direction = -1) +
  labs(title = "Churn Rate by Geography", 
       x = "Geography", y = "Proportion") +
  theme_minimal()



# For CreditScore and Churn
ggplot(df.train, aes(x = CreditScore, fill = factor(Exited))) + 
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Credit Score Density by Churn", 
       x = "Credit Score", y = "Density") +
  theme_bw()

# 'Exited' is the target variable 

# Set a seed for reproducibility
set.seed(101)

# Split the data into training and testing sets
split = sample.split(df.train$Exited, SplitRatio = 0.70)
final.train = subset(df.train, split == TRUE)
final.test = subset(df.train, split == FALSE)

# Train the logistic regression model
final.log.model <- glm(Exited ~ ., family = binomial(link = 'logit'), 
                       data = df.train)

# Summarize the model
summary(final.log.model)

# Predict on the test set
fitted.probabilities <- predict(final.log.model, 
                                newdata = final.test, type = 'response')
fitted.results <- ifelse(fitted.probabilities > 0.5, 1, 0)

# Calculate and print the accuracy
misClasificError <- mean(fitted.results != final.test$Exited)
print(paste('Accuracy:', 1 - misClasificError))

# Create a confusion matrix
confusionMatrix <- table(final.test$Exited, fitted.results)

head(confusionMatrix)


# Convert to factors for confusion matrix calculations
final.test$Exited <- factor(final.test$Exited, levels = c(0, 1))
fitted.results <- factor(fitted.results, levels = c(0, 1))

# Calculate sensitivity and specificity
sensitivity <- sensitivity(confusionMatrix, positive = "1")
specificity <- specificity(confusionMatrix, positive = "1")

# Print the sensitivity and specificity
print(paste('Sensitivity:', sensitivity))
print(paste('Specificity:', specificity))

# Load the new dataset
new_customer_data <- read.csv("NewCustomerDataset-2.csv")

str(new_customer_data)

head(new_customer_data, 5)

summary(new_customer_data)

# Preprocess the data Handling missing values
new_customer_data$EstimatedSalary[is.na(new_customer_data$EstimatedSalary)] <- 
  median(df.train$EstimatedSalary, na.rm = TRUE)

summary(new_customer_data)
str(new_customer_data)


# Remove unnecessary features
new_customer_data <- new_customer_data %>% select(-id, -CustomerId, -Surname)

# Convert categorical variables to factors
new_customer_data$Geography <- as.factor(new_customer_data$Geography)
new_customer_data$Gender <- as.factor(new_customer_data$Gender)
new_customer_data$IsActiveMember <- as.factor(new_customer_data$IsActiveMember)

# Ensure categorical variables are factorized as in the training set
new_customer_data$Geography <- as.factor(new_customer_data$Geography)
new_customer_data$Gender <- as.factor(new_customer_data$Gender)

str(new_customer_data)


# Predict churn
new_customer_data$predicted_churn <- predict(final.log.model, 
                                             newdata = new_customer_data, 
                                             type = 'response')
new_customer_data$predicted_churn <- 
  ifelse(new_customer_data$predicted_churn > 0.5, 1, 0)

# View the predictions
head(new_customer_data$predicted_churn)


