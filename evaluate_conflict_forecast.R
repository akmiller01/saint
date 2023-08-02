list.of.packages <- c("data.table","reshape2", "ggplot2", "scales", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

forecast = fread("outputs/binary_simple_uppsala_replication_forecast.csv")
forecast$y_prob = forecast$y_hat
forecast$y_hat = round(forecast$y_prob)

groupings = read.xlsx("~/git/humanitarian-ssp-projections/WB/CLASS.xlsx")
names(groupings) = c(
  "country.name",
  "iso3",
  "region",
  "income.group",
  "lending.category"
)

forecast = merge(forecast, groupings, by="iso3", all.x=T)

forecast$conflict[which(forecast$year>=2014)] = forecast$y_hat[which(forecast$year>=2014)]
forecast_agg = forecast[,.(
  conflicts=sum(conflict, na.rm=T)
), by=.(Scenario, year)]

ggplot(forecast_agg, aes(x=year,y=conflicts,group=Scenario,color=Scenario)) +
  scale_y_continuous(labels=label_comma()) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Global conflicts")


forecast_agg_region = forecast[,.(
  conflicts=sum(conflict, na.rm=T)
), by=.(Scenario, year, region)]

ggplot(forecast_agg_region, aes(x=year,y=conflicts,group=region,fill=region)) +
  scale_y_continuous(labels=label_comma()) +
  geom_area(stat="identity") +
  facet_grid(Scenario ~ .) +
  theme_classic() +
  labs(x="Year", y="Global conflicts")
