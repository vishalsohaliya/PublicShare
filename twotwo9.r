# Given weights
weights <- c(40, 70, 61, 58, 58, 50, 72, 63, 51, 62,
             65, 60, 68, 68, 78, 54, 52, 60, 50, 70,
             60, 35, 53, 58, 79, 60, 62, 61, 55, 65,
             51, 39, 45, 58, 50, 65, 62, 50, 72, 62,
             52, 65, 67, 87, 45, 75, 71, 52, 65, 59)

# (i) Minimum and Maximum weight
min_weight <- min(weights)
max_weight <- max(weights)
cat("Minimum weight:", min_weight, "\n")
cat("Maximum weight:", max_weight, "\n")

# (ii) Percentage of adults with weight between 50 and 60
count_50_60 <- sum(weights >= 50 & weights <= 60)
total_count <- length(weights)
percentage_50_60 <- (count_50_60 / total_count) * 100
cat("Percentage of adults with weight between 50 and 60 kg:", percentage_50_60, "%\n")

# (iii) Frequency distribution graph
hist(weights, main="Frequency Distribution of Weights", xlab="Weight (kg)", ylab="Frequency", col="blue", breaks=10)


######################################

# Data for interest rates and average loan amounts
loan_data <- data.frame(
  Year = 2006:2021,
  InterestRate = c(10.30, 10.20, 10.10, 9.50, 8.50, 7.40, 8.40, 7.90, 7.60, 7.50, 6.90, 7.40, 8.00, 7.20, 6.50, 6.00),
  LoanAmount = c(174610, 174040, 166155, 164825, 164255, 164540, 164540, 161215, 165775, 169005, 178695, 193040, 218690, 245290, 294310, 313310)
)

# Fit the linear regression model
model <- lm(LoanAmount ~ InterestRate, data = loan_data)

# Display the summary of the model
summary(model)

# Predict loan amount for an interest rate of 5%
new_data <- data.frame(InterestRate = 5)
predicted_loan_amount <- predict(model, newdata = new_data)
cat("Predicted average loan amount for an interest rate of 5%:", predicted_loan_amount, "\n")

# Plot the data points
plot(loan_data$InterestRate, loan_data$LoanAmount, 
     main="Linear Regression: Loan Amount vs. Interest Rate",
     xlab="Interest Rate (%)", ylab="Average Loan Amount (INR)",
     pch=19, col="blue")

# Add the regression line
abline(model, col="red", lwd=2)

# Add the prediction point for an interest rate of 5%
points(5, predicted_loan_amount, col="green", pch=19)


###########################################

# Given weights
weights <- c(60, 75, 63, 55, 88, 65, 72, 75, 88, 63,
             65, 60, 78, 68, 78, 74, 82, 66, 81, 71,
             69, 59, 58, 78, 89, 72, 68, 71, 65, 85,
             76, 77, 69, 56, 64, 76, 55, 74, 68, 76,
             56, 63, 67, 187, 75, 85, 71, 64, 66, 19)

# (iv) Outliers
boxplot(weights, main="Boxplot of Weights", ylab="Weight (kg)", col="red")
outliers <- boxplot.stats(weights)$out
cat("Outliers:", outliers, "\n")

######################################################

# Weight data of 10 students
weights <- c(45, 55, 65, 38, 48, 50, 54, 60, 39, 49)

# Creating a grouped frequency distribution
breaks <- seq(35, 70, by=5)  # Define the breaks for the groups
freq_distribution <- table(cut(weights, breaks))
print(freq_distribution)

# Plotting the frequency distribution
hist(weights, breaks=breaks, main="Frequency Distribution of Weights", xlab="Weight", ylab="Frequency", col="lightblue", border="black")

#########################################################

# Income data of 10 persons
income <- c(25000, 35000, 30000, 45000, 29000, 27000, 47000, 51000, 25000, 39000)

# Define the breaks for the income classes
breaks <- c(20000, 30000, 40000, 50000)

# Create the frequency distribution
income_distribution <- cut(income, breaks, right=FALSE)
frequency <- table(income_distribution)

# Display the frequency distribution
print(frequency)

# Plot the frequency distribution as a bar chart
barplot(frequency, main="Income Frequency Distribution", xlab="Income Range", ylab="Frequency", col="lightblue", border="black")

#################################################################

# Study time and final percentage data
study_time <- c(5, 6, 3, 4, 1, 2, 8, 6, 7, 4)
final_percentage <- c(75, 75, 60, 62, 55, 58, 80, 75, 80, 60)

# Create a data frame
data <- data.frame(study_time, final_percentage)

# Fit the linear regression model
model <- lm(final_percentage ~ study_time, data=data)

# Display the summary of the model
summary(model)

# Predict the final percentage for a study time of 5 hours
new_study_time <- data.frame(study_time=5)
predicted_percentage <- predict(model, new_study_time)

# Display the predicted final percentage
cat("Predicted final percentage for a study time of 5 hours:", predicted_percentage, "%\n")

