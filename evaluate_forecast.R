list.of.packages <- c("data.table","reshape2", "ggplot2", "scales", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

forecast = fread("outputs/regression_iiasa_unhcr_displaced2_forecast.csv")
setnames(forecast, "Region", "iso3")

groupings = read.xlsx("~/git/humanitarian-ssp-projections/WB/CLASS.xlsx")
names(groupings) = c(
  "country.name",
  "iso3",
  "region",
  "income.group",
  "lending.category"
)

forecast = merge(forecast, groupings, by="iso3", all.x=T)

forecast$displaced_persons[which(forecast$year>=2023)] = forecast$y_hat[which(forecast$year>=2023)]
forecast$displaced_persons[which(forecast$displaced_persons<0)] = 0
forecast$displaced_persons = forecast$displaced_persons / 1e6
forecast_agg = forecast[,.(
  displaced_persons=sum(displaced_persons, na.rm=T),
  pop=sum(pop, na.rm=T)
  ), by=.(Scenario, year)]

people = dollar_format(prefix="")

ggplot(forecast_agg, aes(x=year,y=displaced_persons,group=Scenario,color=Scenario)) +
  scale_y_continuous(labels=people) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Displaced persons (millions)")

samples = c("UGA")
forecast_sample = subset(forecast, iso3 %in% samples)
forecast_sample$iso3_Scenario = paste0(forecast_sample$iso3, " - ", forecast_sample$Scenario)
ggplot(forecast_sample, aes(x=year,y=displaced_persons,group=iso3_Scenario,fill=iso3_Scenario)) +
  scale_y_continuous(labels=people) +
  geom_area(stat="identity") +
  theme_classic() +
  labs(x="Year", y="Displaced persons (millions)")

forecast_agg$displacement_pc = forecast_agg$displaced_persons / forecast_agg$pop

ggplot(forecast_agg, aes(x=year,y=displacement_pc,group=Scenario,color=Scenario)) +
  scale_y_continuous(labels=percent) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Displaced persons (%)")

forecast_agg_region = forecast[,.(
  displaced_persons=sum(displaced_persons, na.rm=T),
  pop=sum(pop, na.rm=T)
), by=.(Scenario, year, region)]

ggplot(forecast_agg_region, aes(x=year,y=displaced_persons,group=region,fill=region)) +
  scale_y_continuous(labels=people) +
  geom_area(stat="identity") +
  facet_grid(Scenario ~ .) +
  theme_classic() +
  labs(x="Year", y="Displaced persons (millions)")
