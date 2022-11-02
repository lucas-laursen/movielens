
## Installing packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org") #this is supposed to be part of the tidyverse, but for some reason didn't load properly on my machine...
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(lubridate) 
library(caret)
library(data.table)
library(magrittr)

## Downloading data

options(timeout=1000)
dl <- tempfile() #creates a string as a placeholder name for later use
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl) 

## Storing and wrangling data in a useful format, with richer comments in full report

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp")) 

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres") 

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres)) 

movielens <- left_join(ratings, movies, by = "movieId", copy = TRUE)

movielens <- mutate(movielens, date = as_datetime(timestamp))
movielens <- mutate(movielens, date = round_date(date, unit = "year"))

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

write_csv(edx, "edx.csv", )
write_csv(validation, "validation.csv")

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in 'test' set are also in 'train' set
test <- temp %>% semi_join(train, by = "movieId") %>% semi_join(train, by = "userId")

# Add rows removed from 'test' set back into 'train' set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(test_index, temp, removed)

write_csv(train, "train.csv", )
write_csv(test, "test.csv")

##################################################################
## Only run this code if you have already downloaded and wrangled the data 
## into local files and just want to run the analysis
## Otherwise, skip ahead to Analysis section

#edx <- read.csv("edx.csv", sep = ",")
#validation <- read.csv("validation.csv", sep = ",")
#train <- read.csv("train.csv", sep = ",")
#test <- read.csv("test.csv", sep = ",")

##################################################################

##################################################################
## Analysis
##################################################################

# Defining a function to calculate the root mean square error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Model 0
mu_hat <- mean(train$rating)
naive_rmse <- RMSE(train$rating, mu_hat)

#Target RMSE
target_rmse <- 0.8649

#Making a results table
rmse_results <- tibble(method = "Target RMSE", RMSE = num(target_rmse, sigfig=5), "% relative to RMSE" = num((target_rmse- target_rmse) / (target_rmse),sigfig=3)) %>% add_row(method = "Just the average", RMSE = naive_rmse, "% relative to RMSE" = num((naive_rmse - target_rmse)/(target_rmse)*100, sigfig=3))
rmse_results

#Model 1
rm(mu_hat)
mu <- mean(train$rating) #here we dispense with the _hat notation, following the textbook
rm()
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
movie_rmse <-  RMSE(predicted_ratings, test$rating)

rmse_results <- rmse_results %>% add_row(method = "Mean + Movie Effect Model", RMSE = movie_rmse, "% relative to RMSE" = num((movie_rmse - target_rmse)/(target_rmse)*100, sigfig=3)) 

rmse_results

#Model 2
user_avgs <- train %>% 
  group_by(userId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
user_rmse <- RMSE(predicted_ratings, test$rating)

rmse_results <- rmse_results %>% add_row(method = "Mean + Movie + User Effects Model", RMSE = user_rmse, "% relative to RMSE" = num((user_rmse - target_rmse)/(target_rmse)*100, sigfig=3)) 
rmse_results

#Model 3
superuser_avgs <- train %>% 
  group_by(userId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_v = mean(rating - mu - b_i))
predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(superuser_avgs, by='userId') %>%
  mutate(pred = mu + b_i + coalesce(b_v,0.0)) %>% 
  #coalesce() prevents NAs where test + superuser_avgs don't overlap
  pull(pred)
superuser_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Movie + SuperUser Model", RMSE = superuser_rmse, "% relative to RMSE" = num((superuser_rmse - target_rmse)/(target_rmse)*100, sigfig=3)) 

#Model 4

genres_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u)) 

predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
genres_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Movie + User + Genre Model", RMSE = genres_rmse, "% relative to RMSE" = num((genres_rmse - target_rmse)/(target_rmse)*100, sigfig=3)) 

#Model 5

genres_filt <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>% mutate(n_ratings=n()) %>% filter(n_ratings>=1000) %>%
  summarize(b_h = mean(rating - mu - b_i - b_u)) 

predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_filt, by='genres') %>%
  mutate(pred = mu + b_i + b_u + coalesce(b_h,0.0)) %>% 
  #coalesce() prevents NAs where test + superuser_avgs don't overlap
  pull(pred)
genres_filt_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Movie + User + Filtered Genre Model", RMSE = genres_filt_rmse, "% relative to RMSE" = num((genres_filt_rmse - target_rmse)/(target_rmse)*100, sigfig=3)) 

#Model 6

lambda <- 3
mu <- mean(train$rating)
movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by = "movieId") %>% 
  mutate(pred = mu + b_i) %>% 
  pull(pred)
movies_reg_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Regularized Movie Model", RMSE = movies_reg_rmse, "% relative to RMSE" = num((movies_reg_rmse - target_rmse)/(target_rmse)*100, sigfig=3)) 
rmse_results

#Model 7
lambdas <- seq(0, 10, 0.25)
mu <- mean(train$rating)
just_the_sum <- train %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test$rating))
})
lambdas[which.min(rmses)]
movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambdas[which.min(rmses)]), n_i = n())
predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by = "movieId") %>% 
  mutate(pred = mu + b_i) %>% 
  pull(pred)
movies_cv_reg_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Cross-Validated Regularized Movie Model", RMSE = movies_cv_reg_rmse, "% relative to RMSE" = num((movies_cv_reg_rmse- target_rmse)/(target_rmse)*100, sigfig=3)) 
rmse_results

#Model 8
regularization <- function(lambda, trainset, testset){
  mu <- mean(train$rating)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

#for the following sequence of lambdas--THIS CODE IS SLOW
lambdas <- seq(0, 10, 0.25)
#apply the function defined above to generate the possible RMSEs
rmses <- sapply(lambdas, 
                regularization, 
                trainset = train, 
                testset = test)
#and store the optimal lambda
lambda <- lambdas[which.min(rmses)]

#then use that to find the regularized RMSE for movie + user effects

movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
user_avgs <- train %>% 
  group_by(userId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_u = mean(rating - mu - b_i))

movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
user_reg_avgs <- train %>% 
  group_by(userId) %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(movie_reg_avgs, by= 'movieId') %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)
user_cv_reg_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Cross-Validated Regularized Movie + User Model", RMSE = user_cv_reg_rmse, "% relative to RMSE" = num((user_cv_reg_rmse- target_rmse)/(target_rmse)*100, sigfig=3)) 
rmse_results

#Model 9
###--SLOW CODE 
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, 
                regularization, 
                trainset = edx, 
                testset = validation)
lambda <- lambdas[which.min(rmses)]

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
user_reg_avgs <- edx %>% 
  group_by(userId) %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(movie_reg_avgs, by= 'movieId') %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)
validated_user_cv_reg_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- rmse_results %>% add_row(method = "Above model vs. validation data", RMSE = user_cv_reg_rmse, "% relative to RMSE" = num((user_cv_reg_rmse- target_rmse)/(target_rmse)*100, sigfig=3)) 

#Model 10

genre_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u)) 

regularization <- function(lambda, trainset, testset){
  mu <- mean(train$rating)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_u - b_i - mu)/(n()+lambda))
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    filter(!is.na(b_i), !is.na(b_u), !is.na(b_g)) %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

#for the following sequence of lambdas--THIS CODE IS SLOW
lambdas <- seq(0, 10, 0.25)
#apply the function defined above to generate the possible RMSEs
rmses <- sapply(lambdas, 
                regularization, 
                trainset = train, 
                testset = test)
#and store the optimal lambda
lambda <- lambdas[which.min(rmses)]

movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
user_reg_avgs <- train %>% 
  group_by(userId) %>% 
  left_join(movie_reg_avgs, by= 'movieId') %>%
  left_join(user_avgs, by='userId') %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
genre_reg_avgs <- train %>% 
  group_by(genres) %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(movie_avgs, by= 'movieId') %>%
  left_join(genre_avgs, by= 'genres') %>%
  summarize(b_g = sum(rating - mu - b_u)/(n()+lambda), n_u = n())
predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(genre_reg_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% 
  pull(pred)
genres_cv_reg_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Movie + User + Genre Cross-Validated Regularized Model", RMSE = genres_cv_reg_rmse, "% relative to RMSE" = num((genres_cv_reg_rmse- target_rmse)/(target_rmse)*100, sigfig=3)) 
rmse_results

#Model 11

#adapting code from the genre section to consider the average rating for each year
time_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(date) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u)) 

#adapting the regularization function to include time...
regularization <- function(lambda, trainset, testset){
  mu <- mean(train$rating)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_t <- trainset %>% 
    left_join(b_u, by="userId") %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - b_u - b_i - mu)/(n()+lambda))
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "date") %>%
    filter(!is.na(b_i), !is.na(b_u), !is.na(b_t)) %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

#for the following sequence of lambdas--THIS CODE IS SLOW
lambdas <- seq(0, 10, 0.25)
#apply the function defined above to generate the possible RMSEs
rmses <- sapply(lambdas, 
                regularization, 
                trainset = train, 
                testset = test)
#and store the optimal lambda
lambda <- lambdas[which.min(rmses)]

#then use that to find the regularized RMSE for movie + user + time effects

movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
user_reg_avgs <- train %>% 
  group_by(userId) %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(movie_reg_avgs, by= 'movieId') %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
time_reg_avgs <- train %>% 
  group_by(date) %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(movie_reg_avgs, by= 'movieId') %>%
  left_join(time_avgs, by= 'date') %>%
  summarize(b_t = sum(rating - mu - b_u)/(n()+lambda), n_u = n())
predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(time_reg_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>% 
  pull(pred)
time_cv_reg_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- rmse_results %>% add_row(method = "Mean + Cross-Validated Regularized Movie + User + Time Model", RMSE = time_cv_reg_rmse, "% relative to RMSE" = num((time_cv_reg_rmse- target_rmse)/(target_rmse)*100, sigfig=3)) 
rmse_results

