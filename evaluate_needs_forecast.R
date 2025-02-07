list.of.packages <- c("data.table","reshape2", "ggplot2", "scales", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

forecast = fread("outputs/regression_tripartite_bigram_forecast.csv")
# scenario_split = strsplit(forecast$scenario, split = "|", fixed=T)
# forecast$scenario = sapply(scenario_split, `[[`, 1)
# forecast$iso3 = sapply(scenario_split, `[[`, 2)
# forecast$year = sapply(scenario_split, `[[`, 3)

groupings = read.xlsx("~/git/humanitarian-ssp-projections/WB/CLASS.xlsx")
names(groupings) = c(
  "country.name",
  "iso3",
  "region",
  "income.group",
  "lending.category"
)

forecast = merge(forecast, groupings, by="iso3", all.x=T)

forecast$humanitarian_needs = forecast$y_hat
forecast$humanitarian_needs[which(forecast$humanitarian_needs<0)] = 0
forecast$humanitarian_needs = forecast$humanitarian_needs

forecast_agg = forecast[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  climate_disasters=sum(climate_disasters, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario, year)]

ggplot(forecast_agg, aes(x=year,y=humanitarian_needs,group=scenario,color=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="People in need (millions)")
