list.of.packages <- c("data.table","OpenML", "farff", "ggplot2", "plm", "dplyr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

# $ python train.py --dset_id conflict_clim_bigram --task binary
# $ python sample.py --dset_id conflict_clim_bigram --task binary
saint = fread("~/git/saint/outputs/binary_conflict_clim_bigram.csv")
saint$y_prob = saint$y_hat
saint$y_hat = round(saint$y_prob)
ggplot(saint, aes(x=factor(y), y=y_prob)) + geom_violin(scale="width")
pred = data.table(table(saint$y, saint$y_hat))
names(pred) = c("y", "y_hat", "count")
pred$correct = pred$y == pred$y_hat
recall = pred$count[which(pred$y==1 & pred$y_hat==1)] / sum(pred$count[which(pred$y==1)])
precision = pred$count[which(pred$y==1 & pred$y_hat==1)] / sum(pred$count[which(pred$y_hat==1)])
accuracy = mean(saint$y==saint$y_hat)
print(
  paste0(
    "Recall: ", recall,
    "; Precision: ", precision,
    "; Accuracy: ", accuracy
  )
)
ols_data = fread("~/git/saint/data/conflict_clim_bigram.csv")
ols_data$prec = rowSums(
  ols_data[,c(
    "prec_1",
    "prec_2",
    "prec_3",
    "prec_4",
    "prec_5",
    "prec_6",
    "prec_7",
    "prec_8",
    "prec_9",
    "prec_10",
    "prec_11",
    "prec_12"
  )]
)
ols_data$tmax = pmax(
  ols_data$tmax_1,
  ols_data$tmax_2,
  ols_data$tmax_3,
  ols_data$tmax_4,
  ols_data$tmax_5,
  ols_data$tmax_6,
  ols_data$tmax_7,
  ols_data$tmax_8,
  ols_data$tmax_9,
  ols_data$tmax_10,
  ols_data$tmax_11,
  ols_data$tmax_12
)
fit_simple = glm(
  conflict~
    gdpgrowth+
    prec+
    tmax,
  data=ols_data, family="binomial"
)
summary(fit_simple)
fit = glm(
  conflict~
    gdpgrowth+
    prec_1+
    prec_2+
    prec_3+
    prec_4+
    prec_5+
    prec_6+
    prec_7+
    prec_8+
    prec_9+
    prec_10+
    prec_11+
    prec_12+
    tmax_1+
    tmax_2+
    tmax_3+
    tmax_4+
    tmax_5+
    tmax_6+
    tmax_7+
    tmax_8+
    tmax_9+
    tmax_10+
    tmax_11+
    tmax_12+
    iso3,
  data=ols_data, family="binomial"
)
summary(fit)
nullhypo <- glm(conflict~1, data=ols_data, family="binomial")
mcFadden = 1-logLik(fit)/logLik(nullhypo)
mcFadden
forecast = fread("~/git/saint/data/conflict_clim_forecasting.csv")
forecast$prec = rowSums(
  forecast[,c(
    "prec_1",
    "prec_2",
    "prec_3",
    "prec_4",
    "prec_5",
    "prec_6",
    "prec_7",
    "prec_8",
    "prec_9",
    "prec_10",
    "prec_11",
    "prec_12"
  )]
)
forecast$tmax = pmax(
  forecast$tmax_1,
  forecast$tmax_2,
  forecast$tmax_3,
  forecast$tmax_4,
  forecast$tmax_5,
  forecast$tmax_6,
  forecast$tmax_7,
  forecast$tmax_8,
  forecast$tmax_9,
  forecast$tmax_10,
  forecast$tmax_11,
  forecast$tmax_12
)
forecast$y_prob = predict.glm(fit, newdata = forecast)
forecast$y_prob = forecast$y_prob - min(forecast$y_prob, na.rm=T)
forecast$y_prob = forecast$y_prob / max(forecast$y_prob, na.rm=T)
forecast$y_hat = round(forecast$y_prob)
forecast_pred = subset(forecast, !is.na(conflict))
pred = data.table(table(forecast_pred$conflict, forecast_pred$y_hat))
names(pred) = c("y", "y_hat", "count")
pred$correct = pred$y == pred$y_hat
recall = pred$count[which(pred$y==1 & pred$y_hat==1)] / sum(pred$count[which(pred$y==1)])
precision = pred$count[which(pred$y==1 & pred$y_hat==1)] / sum(pred$count[which(pred$y_hat==1)])
accuracy = mean(forecast_pred$conflict==forecast_pred$y_hat)
print(
  paste0(
    "Recall: ", recall,
    "; Precision: ", precision,
    "; Accuracy: ", accuracy
  )
)
forecast$conflict[which(forecast$year>=2014)] = forecast$y_hat[which(forecast$year>=2014)]
forecast_agg = forecast[,.(
  conflicts=sum(y_hat, na.rm=T)
), by=.(scenario, year)]
library(scales)
ggplot(forecast_agg, aes(x=year,y=conflicts,group=scenario,color=scenario)) +
  scale_y_continuous(labels=label_comma()) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Global conflicts")


# $ python train.py --dset_id displacement_worldclim --task regression --embedding_size=8 --epochs=1000
# $ python sample.py --dset_id displacement_worldclim --task regression --embedding_size=8 --epochs=1000
saint = fread("~/git/saint/outputs/regression_displacement_worldclim.csv")
(sum(saint$y_hat)-sum(saint$y))/sum(saint$y)
plot(saint)
summary(lm(y~y_hat, data=saint))
ols_data = fread("~/git/saint/data/displacement_worldclim.csv")
ols_data$prec = rowSums(
  ols_data[,c(
    "prec_1",
    "prec_2",
    "prec_3",
    "prec_4",
    "prec_5",
    "prec_6",
    "prec_7",
    "prec_8",
    "prec_9",
    "prec_10",
    "prec_11",
    "prec_12"
  )]
)
ols_data$tmax = pmax(
  ols_data$tmax_1,
  ols_data$tmax_2,
  ols_data$tmax_3,
  ols_data$tmax_4,
  ols_data$tmax_5,
  ols_data$tmax_6,
  ols_data$tmax_7,
  ols_data$tmax_8,
  ols_data$tmax_9,
  ols_data$tmax_10,
  ols_data$tmax_11,
  ols_data$tmax_12
)
ols_simple = lm(
  displaced_persons~
    gdpgrowth+
    prec+tmax,
  data=ols_data
)
summary(ols_simple)
ols = lm(
  displaced_persons~
    gdpgrowth+
    prec_1+
    prec_2+
    prec_3+
    prec_4+
    prec_5+
    prec_6+
    prec_7+
    prec_8+
    prec_9+
    prec_10+
    prec_11+
    prec_12+
    tmax_1+
    tmax_2+
    tmax_3+
    tmax_4+
    tmax_5+
    tmax_6+
    tmax_7+
    tmax_8+
    tmax_9+
    tmax_10+
    tmax_11+
    tmax_12+
    iso3,
  data=ols_data
)
summary(ols)
forecast = fread("~/git/saint/data/displacement_worldclim_forecasting.csv")
forecast$prec = rowSums(
  forecast[,c(
    "prec_1",
    "prec_2",
    "prec_3",
    "prec_4",
    "prec_5",
    "prec_6",
    "prec_7",
    "prec_8",
    "prec_9",
    "prec_10",
    "prec_11",
    "prec_12"
  )]
)
forecast$tmax = pmax(
  forecast$tmax_1,
  forecast$tmax_2,
  forecast$tmax_3,
  forecast$tmax_4,
  forecast$tmax_5,
  forecast$tmax_6,
  forecast$tmax_7,
  forecast$tmax_8,
  forecast$tmax_9,
  forecast$tmax_10,
  forecast$tmax_11,
  forecast$tmax_12
)
forecast$y_hat = predict.lm(ols, newdata = forecast)
forecast_sub = subset(forecast, !is.na(displaced_persons))
(sum(forecast_sub$y_hat)-sum(forecast_sub$displaced_persons))/sum(forecast_sub$displaced_persons)
forecast$displaced_persons[which(forecast$year>=2023)] = forecast$y_hat[which(forecast$year>=2023)]
forecast_agg = forecast[,.(
  displaced_persons=sum(y_hat, na.rm=T),
  gdpgrowth = mean(gdpgrowth, na.rm=T)
), by=.(scenario, year)]
library(scales)
ggplot(forecast_agg, aes(x=year,y=displaced_persons,group=scenario,color=scenario)) +
  scale_y_continuous(labels=label_comma()) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Displaced persons")


# $ python train.py --dset_id displacement_worldclim2 --task regression --embedding_size=8 --epochs=1000
# $ python sample.py --dset_id displacement_worldclim2 --task regression --embedding_size=8 --epochs=1000
saint = fread("~/git/saint/outputs/regression_displacement_worldclim2.csv")
(sum(saint$y_hat)-sum(saint$y))/sum(saint$y)
plot(saint)
summary(lm(y~y_hat, data=saint))
ols_data = fread("~/git/saint/data/displacement_worldclim2.csv")
ols_data$prec = rowSums(
  ols_data[,c(
    "prec_1",
    "prec_2",
    "prec_3",
    "prec_4",
    "prec_5",
    "prec_6",
    "prec_7",
    "prec_8",
    "prec_9",
    "prec_10",
    "prec_11",
    "prec_12"
  )]
)
ols_data$tmax = pmax(
  ols_data$tmax_1,
  ols_data$tmax_2,
  ols_data$tmax_3,
  ols_data$tmax_4,
  ols_data$tmax_5,
  ols_data$tmax_6,
  ols_data$tmax_7,
  ols_data$tmax_8,
  ols_data$tmax_9,
  ols_data$tmax_10,
  ols_data$tmax_11,
  ols_data$tmax_12
)
ols_simple = lm(
  displaced_persons~
    prec+tmax,
  data=ols_data
)
summary(ols_simple)
ols = lm(
  displaced_persons~
    prec_1+
    prec_2+
    prec_3+
    prec_4+
    prec_5+
    prec_6+
    prec_7+
    prec_8+
    prec_9+
    prec_10+
    prec_11+
    prec_12+
    tmax_1+
    tmax_2+
    tmax_3+
    tmax_4+
    tmax_5+
    tmax_6+
    tmax_7+
    tmax_8+
    tmax_9+
    tmax_10+
    tmax_11+
    tmax_12+
    iso3,
  data=ols_data
)
summary(ols)


# $ python train.py --dset_id climate_worldclim --task regression
# $ python sample.py --dset_id climate_worldclim --task regression
saint = fread("~/git/saint/outputs/regression_climate_worldclim.csv")
(sum(saint$y_hat)-sum(saint$y))/sum(saint$y)
plot(saint)
summary(lm(y~y_hat, data=saint))
ols_data = fread("~/git/saint/data/climate_worldclim.csv")
ols_data$prec = rowSums(
  ols_data[,c(
    "prec_1",
      "prec_2",
      "prec_3",
      "prec_4",
      "prec_5",
      "prec_6",
      "prec_7",
      "prec_8",
      "prec_9",
      "prec_10",
      "prec_11",
      "prec_12"
  )]
)
ols_data$tmax = pmax(
  ols_data$tmax_1,
  ols_data$tmax_2,
  ols_data$tmax_3,
  ols_data$tmax_4,
  ols_data$tmax_5,
  ols_data$tmax_6,
  ols_data$tmax_7,
  ols_data$tmax_8,
  ols_data$tmax_9,
  ols_data$tmax_10,
  ols_data$tmax_11,
  ols_data$tmax_12
)
ols_simple = lm(
  max_affected~
    prec+tmax,
  data=ols_data
)
summary(ols_simple)
ols = lm(
  max_affected~
    prec_1+
    prec_2+
    prec_3+
    prec_4+
    prec_5+
    prec_6+
    prec_7+
    prec_8+
    prec_9+
    prec_10+
    prec_11+
    prec_12+
    tmax_1+
    tmax_2+
    tmax_3+
    tmax_4+
    tmax_5+
    tmax_6+
    tmax_7+
    tmax_8+
    tmax_9+
    tmax_10+
    tmax_11+
    tmax_12+
    iso3,
  data=ols_data
)
summary(ols)
forecast = fread("~/git/saint/data/climate_worldclim_forecasting.csv")
forecast$prec = rowSums(
  forecast[,c(
    "prec_1",
    "prec_2",
    "prec_3",
    "prec_4",
    "prec_5",
    "prec_6",
    "prec_7",
    "prec_8",
    "prec_9",
    "prec_10",
    "prec_11",
    "prec_12"
  )]
)
forecast$tmax = pmax(
  forecast$tmax_1,
  forecast$tmax_2,
  forecast$tmax_3,
  forecast$tmax_4,
  forecast$tmax_5,
  forecast$tmax_6,
  forecast$tmax_7,
  forecast$tmax_8,
  forecast$tmax_9,
  forecast$tmax_10,
  forecast$tmax_11,
  forecast$tmax_12
)
forecast$y_hat = predict.lm(ols, newdata = forecast)
forecast_sub = subset(forecast, year < 2023 & scenario=="ssp1")
(sum(forecast_sub$y_hat)-sum(forecast_sub$climate_disasters))/sum(forecast_sub$climate_disasters)
forecast$climate_disasters[which(forecast$year>=2023)] = forecast$y_hat[which(forecast$year>=2023)]
forecast_agg = forecast[,.(
  climate_disasters=sum(climate_disasters, na.rm=T)
), by=.(scenario, year)]
library(scales)
ggplot(forecast_agg, aes(x=year,y=climate_disasters,group=scenario,color=scenario)) +
  scale_y_continuous(labels=label_comma()) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Climate disasters")


# $ python train.py --dset_id tripartite_bigram --task regression --embedding_size=8 --epochs 1000
# $ python sample.py --dset_id tripartite_bigram --task regression --embedding_size=8 --epochs 1000
saint = fread("~/git/saint/outputs/regression_tripartite_bigram.csv")
(sum(saint$y_hat)-sum(saint$y))/sum(saint$y)
plot(saint)
summary(lm(y~y_hat, data=saint))
ols_data = fread("~/git/saint/data/tripartite_bigram.csv")
ols_data = ols_data[order(ols_data$iso3, ols_data$year),]
ols_data <- ols_data %>%                           
  group_by(iso3) %>%
  dplyr::mutate(
    hum_t1 = lag(humanitarian_needs, n = 1, default = 0)
  )
ols = lm(humanitarian_needs~
           # hum_t1+
           displaced_persons+
           climate_disasters+
           conflict
         , data=ols_data
)
summary(ols)
forecast = fread("~/git/saint/data/tripartite_bigram_forecasting.csv")
forecast = forecast[order(forecast$scenario, forecast$iso3, forecast$year),]
# forecast <- forecast %>%                           
#   group_by(iso3) %>%
#   dplyr::mutate(
#     hum_t1 = lag(humanitarian_needs, n = 1, default = 0)
#   )
# 
# pb = txtProgressBar(min=2023, max=2100, style=3)
# for(year in c(2023:2100)){
#   setTxtProgressBar(pb, year)
#   forecast$humanitarian_needs[which(forecast$year==year)] =
#     predict.lm(ols, newdata = forecast[which(forecast$year==year),])
#   forecast <- forecast %>%                           
#     group_by(iso3) %>%
#     dplyr::mutate(
#       hum_t1 = lag(humanitarian_needs, n = 1, default = 0)
#     )
# }
# close(pb)
forecast$y_hat = predict.lm(ols, newdata=forecast)
forecast_sub = subset(forecast, year < 2023)
(sum(forecast_sub$y_hat)-sum(forecast_sub$humanitarian_needs))/sum(forecast_sub$humanitarian_needs)

forecast$humanitarian_needs = forecast$y_hat
forecast$scenario = toupper(forecast$scenario)
forecast_sub = subset(forecast, year > 2023 & year < 2101)
forecast_agg = data.table(forecast_sub)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario)]
library(scales)
ggplot(forecast_agg, aes(x=scenario,y=humanitarian_needs,fill=scenario)) +
  scale_y_continuous(labels=label_comma()) +
  geom_bar(stat="identity", position="dodge") +
  theme_classic() +
  labs(x="SSP Scenario", y="People in need (millions)",
       title="Projected global people in humanitarian need (2024-2100)")
forecast_agg = data.table(forecast)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  climate_disasters=sum(climate_disasters, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario, year)]
forecast_agg_baseline = forecast_agg$humanitarian_needs[which.min(forecast_agg$humanitarian_needs)]
forecast_agg$humanitarian_needs = forecast_agg$humanitarian_needs / forecast_agg_baseline
ggplot(forecast_agg, aes(x=year,y=humanitarian_needs,group=scenario,color=scenario)) +
  scale_y_continuous(labels=percent) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="People in need (% of baseline)")

fts = fread("~/git/humanitarian-ssp-projections/fts/hum_reqs.csv")
fts = subset(fts, type!="Regional response plan")
keep = c("iso3", "year", "Total current requirements:")
fts = fts[,keep, with=F]
setnames(fts, "Total current requirements:", "humanitarian_needs")
fts$humanitarian_needs = as.numeric(fts$humanitarian_needs)
rrps = fread("~/git/humanitarian-ssp-projections/fts/hum_reqs_rrps.csv")
keep = c("iso3", "year", "Current reqs.")
rrps = rrps[,keep,with=F]
setnames(rrps, "Current reqs.", "humanitarian_needs")
rrps$humanitarian_needs = as.numeric(gsub(",","",rrps$humanitarian_needs))
fts = rbind(fts, rrps)
fts = subset(fts, nchar(iso3) == 3 & !is.na(humanitarian_needs))
fts_aggregate = fts[,.(humanitarian_needs=sum(humanitarian_needs)),by=.(iso3, year)]

setnames(ols_data, "humanitarian_needs", "pin")
ols_data = merge(ols_data, fts_aggregate, by=c("iso3", "year"))
ols = lm(humanitarian_needs~
           # hum_t1+
           pin
         , data=ols_data
)
summary(ols)
setnames(forecast, "humanitarian_needs", "pin")
forecast$humanitarian_needs = predict.lm(ols, newdata=forecast)
forecast_agg = data.table(forecast)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  pin=sum(pin, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  climate_disasters=sum(climate_disasters, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario, year)]
forecast_agg$humanitarian_needs = forecast_agg$humanitarian_needs / 1e9
ggplot(forecast_agg, aes(x=year,y=humanitarian_needs,group=scenario,color=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Humanitarian needs (billions USD)")
forecast_sub = subset(forecast, year > 2023)
forecast_agg = data.table(forecast_sub)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  pin=sum(pin, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario)]
forecast_agg$humanitarian_needs = forecast_agg$humanitarian_needs / 1e9
ggplot(forecast_agg, aes(x=scenario,y=humanitarian_needs,fill=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_bar(stat="identity", position="dodge") +
  theme_classic() +
  labs(x="SSP Scenario", y="Humanitarian needs (billions USD)",
       title="Projected global humanitarian needs (2024-2100)")
