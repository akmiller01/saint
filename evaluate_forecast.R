list.of.packages <- c("data.table","reshape2", "ggplot2", "scales")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

forecast = fread("outputs/regression_iiasa_unhcr_displaced2_bigram_forecast.csv")

forecast$displaced_persons[which(forecast$year>=2023)] = forecast$y_hat[which(forecast$year>=2023)]
forecast$displaced_persons[which(forecast$displaced_persons<0)] = 0
forecast_agg = forecast[,.(displaced_persons=sum(displaced_persons, na.rm=T)/1e6), by=.(Scenario, year)]

people = dollar_format(prefix="")

ggplot(forecast_agg, aes(x=year,y=displaced_persons,group=Scenario,color=Scenario)) +
  scale_y_continuous(labels=people) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Displaced persons (millions)")
