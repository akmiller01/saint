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

forecast$humanitarian_needs[which(forecast$year>=2023)] = forecast$y_hat[which(forecast$year>=2023)]
forecast$humanitarian_needs[which(forecast$humanitarian_needs<0)] = 0
forecast$humanitarian_needs = forecast$humanitarian_needs / 1e6

forecast_agg_scen = subset(forecast, year>2022 & year<2101)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  climate_disasters=sum(climate_disasters, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario)]
ggplot(forecast_agg_scen, aes(x=scenario,y=humanitarian_needs,fill=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_bar(stat="identity", position="dodge") +
  theme_classic() +
  labs(x="SSP Scenario", y="Humanitarian needs (million USD$)",
       title="Projected global humanitarian needs (2023-2100)")

forecast_agg = forecast[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  climate_disasters=sum(climate_disasters, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario, year)]
forecast_agg_l = melt(forecast_agg, id.vars=c("scenario", "year"))
ggplot(subset(forecast_agg_l, variable=="conflict"), aes(x=year,y=value,group=variable,color=variable)) +
  geom_line() +
  facet_grid(scenario ~.)

ggplot(forecast_agg, aes(x=year,y=humanitarian_needs,group=scenario,color=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Humanitarian needs (millions USD$)")
