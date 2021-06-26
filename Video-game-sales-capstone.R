###################################################
######## Video games sales predictor model ########
###################################################

# Install packages automatically if required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# Load required packages
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(knitr)
library(stringr)
library(lubridate)
library(tinytex)
library(RCurl)
library(randomForest)

# Provide relative path to dataset
game_sales <- read.csv("https://github.com/lowy0047/Video-game-sales-capstone/blob/main/Video_Games_Sales_as_at_22_Dec_2016.csv?raw=true")
game_sales <- na.omit(game_sales)

# Keep only global_sales column
game_sales <- subset(game_sales, select = -c(NA_Sales, EU_Sales, JP_Sales, Other_Sales))

# Remove tbd from user scores and convert column to integer class
game_sales$User_Score <- gsub('tbd', '', game_sales$User_Score)
game_sales$User_Score <- as.numeric(game_sales$User_Score)
summary(game_sales)

# Set seed to 1 then split data into 80/20 training/test set
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = game_sales$Global_Sales, times = 1, 
                                  p = 0.2, list = FALSE)
train_set <- game_sales[-test_index,]
test_set <- game_sales[test_index,]

# Plot average sales vs critic score
train_set %>% group_by(Critic_Score) %>%
  summarize(avg_sales = mean(Global_Sales)) %>%
  ggplot(aes(Critic_Score, avg_sales)) +
  geom_point() +
  labs(title = "Video games average sales vs critic score", x = "Critic score", 
       y = "Average sales")

# Plot average sales vs user score
train_set %>% group_by(User_Score) %>%
  summarize(avg_sales = mean(Global_Sales)) %>%
  ggplot(aes(User_Score, avg_sales)) +
  geom_point() +
  labs(title = "Video games average sales vs user score", x = "User score", 
       y = "Average sales")

# Plot spread of global sales vs genre
options(scipen=999) # Remove scientific notation
ggplot(train_set, aes(Genre, Global_Sales)) +
  geom_boxplot() +
  scale_y_log10() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Video games global sales spread vs genre", x = "Genre", 
           y = "Global sales")

# Plot combined global sales vs genre
train_set %>% 
  ggplot(aes(Genre)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Video games combined global sales vs Genre", x = "Genre", 
       y = "Combined global sales")

# Plot spread of global sales vs platform
ggplot(train_set, aes(Platform, Global_Sales)) +
  geom_boxplot() +
  scale_y_log10() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Video games global sales spread vs platform", x = "Platform", 
       y = "Global sales")

# Plot combined global sales vs platform
train_set %>% 
  ggplot(aes(Platform)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Video games combined global sales vs platform", x = "Platform", 
       y = "Combined global sales")

# Baseline model: Average of global sales
mu <- mean(train_set$Global_Sales)
base_rmse <- RMSE(test_set$Global_Sales, mu)
rmse_table <- tibble(Method = "Baseline average model", RMSE = base_rmse)
knitr::kable(rmse_table)

# Model 1: Linear regression modeling critic score
model1 <- lm(Global_Sales~Critic_Score, data = train_set)
predicted <- predict(model1, test_set)
model1rmse <- RMSE(predicted, test_set$Global_Sales)
rmse_table <- rbind(rmse_table, tibble(Method = "Critic score linear model", 
                                       RMSE = model1rmse))
knitr::kable(rmse_table)

# Model 2: Linear regression modeling critic and user scores
model2 <- lm(Global_Sales~Critic_Score+User_Score, data = train_set)
predicted2 <- predict(model2, test_set)
model2rmse <- RMSE(predicted2, test_set$Global_Sales)
rmse_table <- rbind(rmse_table, tibble(Method = "Critic & user scores linear model", 
                                       RMSE = model2rmse))
knitr::kable(rmse_table)

# Model 3: Linear regression modeling platform, critic and user scores
model3 <- lm(Global_Sales~Critic_Score+User_Score+Platform, data = train_set)
predicted3 <- predict(model3, test_set)
model3rmse <- RMSE(predicted3, test_set$Global_Sales)
rmse_table <- rbind(rmse_table, 
                    tibble(Method = "Platform, critic & user scores linear model", 
                           RMSE = model3rmse))
knitr::kable(rmse_table)

# Model 4: Linear regression modeling genre, platform, critic and user scores
model4 <- lm(Global_Sales~Critic_Score+User_Score+Platform+Genre, data = train_set)
predicted4 <- predict(model4, test_set)
model4rmse <- RMSE(predicted4, test_set$Global_Sales)
rmse_table <- rbind(rmse_table, tibble(Method = "Genre, platform, critic & 
                                       user scores linear model", RMSE = model4rmse))
knitr::kable(rmse_table)

# Model 5: k-Nearest neighbors
model5 <- knnreg(Global_Sales~Critic_Score+User_Score+Platform+Genre, 
                       data = train_set)
predicted5 <- predict(model5, test_set)
model5rmse <- RMSE(predicted5, test_set$Global_Sales)
rmse_table <- rbind(rmse_table, tibble(Method = "Genre, platform, critic & 
                                       user scores kNN model", RMSE = model5rmse))
knitr::kable(rmse_table)

# Model 6: Random forests
model6 <- randomForest(Global_Sales~Critic_Score+User_Score+Platform+Genre, 
                 data = train_set)
predicted6 <- predict(model6, test_set)
model6rmse <- RMSE(predicted6, test_set$Global_Sales)
rmse_table <- rbind(rmse_table, tibble(Method = "Genre, platform, critic & 
                                       user scores random forest model", 
                                       RMSE = model5rmse))
knitr::kable(rmse_table)

