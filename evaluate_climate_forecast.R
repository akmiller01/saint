list.of.packages <- c("data.table","reshape2", "ggplot2", "scales", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

forecast = fread("outputs/regression_climate_worldclim_forecast.csv")
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

forecast$climate_disasters[which(forecast$year>=2013)] = forecast$y_hat[which(forecast$year>=2013)]
forecast$climate_disasters[which(forecast$climate_disasters<0)] = 0

forecast_agg_scen = subset(forecast,year>=2013)[,.(
  climate_disasters=sum(climate_disasters, na.rm=T),
  prec_1=mean(prec_1, na.rm=T),
  tmax_1=mean(tmax_1, na.rm=T)
), by=.(scenario)]
ggplot(forecast_agg_scen, aes(x=scenario,y=climate_disasters,fill=scenario)) +
  scale_y_continuous() +
  geom_bar(stat="identity", position="dodge") +
  theme_classic()

forecast_agg = forecast[,.(
  climate_disasters=sum(climate_disasters, na.rm=T),
  prec_1=mean(prec_1, na.rm=T),
  tmax_1=mean(tmax_1, na.rm=T)
), by=.(scenario, year)]
forecast_agg_l = melt(forecast_agg, id.vars=c("scenario", "year"))
ggplot(forecast_agg_l, aes(x=year,y=value,group=variable,color=variable)) +
  geom_line() +
  facet_grid(scenario ~.)

ggplot(forecast_agg, aes(x=year,y=climate_disasters,group=scenario,color=scenario)) +
  scale_y_continuous() +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Total global climate disasters")

