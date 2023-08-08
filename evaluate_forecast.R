list.of.packages <- c("data.table","reshape2", "ggplot2", "scales", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

forecast = fread("outputs/regression_displacement_worldclim_forecast.csv")

historical = subset(forecast, year<2023)
historical = historical[,c("scenario", "iso3", "year", "y_hat")]
historical_w = dcast(historical, iso3+year~scenario, value.var="y_hat")

groupings = read.xlsx("~/git/humanitarian-ssp-projections/WB/CLASS.xlsx")
names(groupings) = c(
  "country.name",
  "iso3",
  "region",
  "income.group",
  "lending.category"
)

forecast = merge(forecast, groupings, by="iso3", all.x=T)

forecast$displaced_persons[which(forecast$year>=1960)] = forecast$y_hat[which(forecast$year>=1960)]
forecast$displaced_persons[which(forecast$displaced_persons<0)] = 0
forecast$displaced_persons = forecast$displaced_persons / 1e6
forecast_agg = forecast[,.(
  displaced_persons=sum(displaced_persons, na.rm=T)
  ), by=.(scenario, year)]

people = dollar_format(prefix="")

ggplot(forecast_agg, aes(x=year,y=displaced_persons,group=scenario,color=scenario)) +
  scale_y_continuous(labels=people) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Displaced persons (millions)")


forecast_agg_region = forecast[,.(
  displaced_persons=sum(displaced_persons, na.rm=T)
), by=.(scenario, year, region)]

ggplot(forecast_agg_region, aes(x=year,y=displaced_persons,group=region,fill=region)) +
  scale_y_continuous(labels=people) +
  geom_area(stat="identity") +
  facet_grid(scenario ~ .) +
  theme_classic() +
  labs(x="Year", y="Displaced persons (millions)")
