#forcasting
#serie 1


#library(fpp3)
library(tsibbledata)
library(flextable)
library(readxl)
library(tidyverse)

# Load necessary libraries
library(tsibbledata)
library(ggplot2)


# Load data sets
data("gafa_stock")
data("PBS")
data("vic_elec")
data("pelt")

# Visualize time series using autoplot from ggfortify
autoplot(gafa_stock) + ggtitle("Time series of GAFA stock data")
autoplot(PBS) + ggtitle("Time series of PBS data")
autoplot(vic_elec) + ggtitle("Time series of Victorian electricity demand data")
autoplot(pelt) + ggtitle("Time series of Canadian pelt data")

#ex2 
a10 <- PBS %>%
  filter(ATC2 == "A10") %>%
  select(Month, Concession, Type, Cost) %>%
  summarise(Cost = sum(Cost)) %>%
  mutate(Cost = Cost / 1e6)

a10.past <- a10 %>% 
  filter(year(Month) < 2006)
a10.future <- a10 %>% 
  filter(year(Month) >= 2006)

#cutting the time series

#You can see that that a10.future has 30 observations. Therefore, create a forecast using the ETS() model
#with automatic model selection for the next 30 periods based on a10.past . Then calculate the accuracy
#scores for these 30 forecasts:
#  Interpret the MAE: is it big or small? Create a table that shows the real costs and the forecasts, and also do
#this graphically.


# Repeat for other loaded packages if necessary
install.packages("fable")

library(tsibble)
library(fable)
a10.fit <- a10.past %>% 
  model(ETS(Cost))
a10.fct <- a10.fit %>% 
  forecast(h = 30)
accuracy(a10.fct, a10.future) 


# here are summmary statistics of the of the costs for a10.past
summary(a10.past$Cost)
sd(a10.past$Cost)

#MAE
library(fable)

# Assuming you have forecasted values stored in a10.fct and actual values in a10.future

# Calculate accuracy
accuracy_metrics <- accuracy(a10.fct, a10.future)

# Extract MAE
mae <- accuracy_metrics$MAE

# View MAE
mae

# the MAE is 2.08, which is roughly 50% of the standard deviation. This seems to indicate that the forecast is quite good.

#Here is a dataframe with real values and forecasts:
# Combine the forecasted values with the actual data
fct.comparison <- data.frame(Period = a10.future$Month,
                             Cost = a10.future$Cost,
                             Cost_Fct = a10.fct$.mean)
names(fct.comparison) <- c("Period", "Cost", "Cost_Fct")
# Calculate the error between forecasted and actual values

fct.comparison <- fct.comparison %>% 
  mutate(Error = Cost_Fct - Cost)
fct.comparison

# Plot the forecast and actual data to be able to visualize
autoplot(a10.fct, level = NULL) +
  geom_line(data = fct.comparison, aes(x = Period, y = Cost_Fct, color = "Forecast")) +
  geom_point(data = fct.comparison, aes(x = Period, y = Cost, color = "Actual")) +
  labs(color = "Data") +
  theme_minimal()

#ex3
#What happens if data are missing? Delete the data from 2000 in the `a10` time series and create a chart of the time series. What do you observe? Then create a forecast model using the same code as shown in Exercise 1. What message do you get?
  
library(dplyr)

a10.subset <- a10 %>% 
  filter(Month < yearmonth("2000-01-01") |
           Month >= yearmonth("2001-01-01"))

a10.subset %>% autoplot(Cost)

a10.subset.fit <-a10.subset %>%  model(M1 = ETS(Cost)
                                       
#We can observe that when there are mssing values, il y a un trait

#Erreur : symbole inattendu dans :
#"a10.subset.fit <-a10.subset %>%  model(M1 = ETS(Cost)
#a10.subset.fit
#on recoit ce message car les % indique qu'on ne forcaste pas 
#L'opérateur %>% (pipe) attend une expression après lui, mais dans votre commande, il y a un espace entre %>% et model(), ce qui provoque une erreur.

#ex4
#the function fill_gap
a10.fill <- a10.subset %>% fill_gaps()
a10.fill %>% autoplot(Cost)

#permet de supprimer le trait des missing values, mtn on voit un trou

#What message do you get when you try to create a model on the time series without data in 2000?
a10.fill.fit <- a10.fill %>% 
  model(
    M1 = ETS(Cost)
  )

#Message d'avis :
#1 error encountered for M1
#[1] ETS does not support missing values.
#Indique que la méthode de lissage exponentiel utilisée dans la fonction ETS() ne prend pas en charge les valeurs manquantes dans les données

#ex5

#1)Count the number of unique values per Material. What do you observe?

sales <-("Sales_Data")
sales.stats <- sale %>% 
  group_by(Material) %>% 
  summarize(n = n()) 
sales.stats
#il y a 5663 unique values per materials

#2)Convert the tibble into a tsibble. Has this tsibble gaps?

# Create the tsibble

# Create a tsibble with Year_Month as the index
library(dplyr)
library(zoo)
library(tsibble)


# Convertir 'Year_Month' en classe 'Date'
sales <- sales %>% 
  mutate(Year_Month = as.Date(Year_Month, format = "%Y-%m"))

# Créer le tsibble avec 'Year_Month' comme index
sales.ts <- sales %>% 
  as_tsibble(index = Year_Month,
             key = c(Business, Group, Material))


# Filling the gaps with 0 and expand to entire time range

sales.gaps <- has_gaps(sales.ts)
sales.gaps <- scan_gaps(sales.ts)

sales.ts <- sales.ts %>% 
  fill_gaps(.full = TRUE, Sales = 0L)

# Create the totals and the add the number of values
# Note that the we re-convert the tsibble into a tibble

sales.stats <- sales.ts %>% 
  as_tibble() %>% 
  group_by(Business, Group, Material) %>% 
  summarize(n = n(),
            Total = sum(Sales)) %>% 
  ungroup() %>% 
  arrange(desc(Total))

# Get the top 12, filter and chart using facet_wrap

top.12 <- sales.stats$Material[1:12]

sales.ts.top.12 <- sales.ts %>% 
  filter(Material %in% top.12)

sales.ts.top.12 %>% autoplot(Sales) + 
  facet_wrap(~ Material, scales = "free_y") +
  theme(legend.position = "none") 

sales.ts %>% filter(Material == "M5453") %>% autoplot()



