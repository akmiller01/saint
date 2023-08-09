list.of.packages <- c("data.table","OpenML", "farff", "ggplot2", "plm")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd_base = "~/git/"
setwd(paste0(wd_base, "saint"))

# $ python train.py --dset_id 44138 --task regression
# $ python sample.py --dset_id 44138 --task regression
saint = fread("~/git/saint/outputs/regression_44138.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))

houses = getOMLDataSet(data.id = 44138L)
houses_data = houses$data
ols = lm(medianhousevalue~
           median_income+
           housing_median_age+
           total_rooms+
           total_bedrooms+
           population+
           households+
           latitude+
           longitude, data=houses_data
)
summary(ols)

# $ python train.py --dset_id ssp1_ext --task multiclass --pretrain
# $ python sample.py --dset_id ssp1_ext --task multiclass
saint = fread("~/git/saint/outputs/multiclass_ssp1_ext.csv")
heatmap(table(saint$y_hat, saint$y), Rowv = NA, Colv = NA, revC = T, scale = "row")
pred = data.table(table(saint$y, saint$y_hat))
names(pred) = c("y", "y_hat", "count")
pred$correct = pred$y == pred$y_hat
View(pred)

# $ python train.py --dset_id ssp1_binary --task binary --pretrain
# $ python sample.py --dset_id ssp1_binary --task binary
saint = fread("~/git/saint/outputs/binary_ssp1_binary.csv")
heatmap(table(saint$y_hat, saint$y), Rowv = NA, Colv = NA, revC = T, scale = "row")
pred = data.table(table(saint$y, saint$y_hat))
names(pred) = c("y", "y_hat", "count")
pred$correct = pred$y == pred$y_hat
View(pred)

# $ python train.py --dset_id data_leak_test --task binary --pretrain
# $ python sample.py --dset_id data_leak_test --task binary
saint = fread("~/git/saint/outputs/binary_data_leak_test.csv")
heatmap(table(saint$y_hat, saint$y), Rowv = NA, Colv = NA, revC = T, scale = "row")
pred = data.table(table(saint$y, saint$y_hat))
names(pred) = c("y", "y_hat", "count")
pred$correct = pred$y == pred$y_hat
View(pred)

# $ python train.py --dset_id ssp1_regression --task regression --pretrain
# $ python sample.py --dset_id ssp1_regression --task regression
saint = fread("~/git/saint/outputs/regression_ssp1_regression.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))

# $ python train.py --dset_id ssp1_fts_all --task regression --pretrain
# $ python sample.py --dset_id ssp1_fts_all --task regression
saint = fread("~/git/saint/outputs/regression_ssp1_fts_all.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))

self_attention_test = data.frame(x=seq(0, 9.999, 0.001))
exponential = sample(c(1:nrow(self_attention_test)), size=nrow(self_attention_test)/2)
self_attention_test$y = (self_attention_test$x * -5) - 10
self_attention_test$formula = "(x * -5) - 10"
self_attention_test$y[exponential] = self_attention_test$x[exponential]^2
self_attention_test$formula[exponential] = "x^2"
self_attention_test = self_attention_test[,c("y","x","formula")]
fwrite(self_attention_test, "~/git/saint/data/self_attention_test.csv")
par(mfrow=c(1,2))
plot(y~x, data=self_attention_test)
ols = lm(y~x+factor(formula), data=self_attention_test)
summary(ols)

# $ python train.py --dset_id self_attention_test --task regression --pretrain
# $ python sample.py --dset_id self_attention_test --task regression # With mod to output x
saint = fread("~/git/saint/outputs/regression_self_attention_test.csv")
plot(y_hat~x,data=saint)
summary(lm(y~y_hat, data=saint))
dev.off()

# $ python train.py --dset_id iiasa_unhcr_refugees --task regression --epochs 500
# $ python sample.py --dset_id iiasa_unhcr_refugees --task regression
saint = fread("~/git/saint/outputs/regression_iiasa_unhcr_refugees.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))
refugee_data = fread("~/git/saint/data/iiasa_unhcr_refugees.csv")
ols = lm(refugees~
           Region+
           year+
           pop+
           gdp+
           urban, data=refugee_data
)
summary(ols)
crd = refugee_data[complete.cases(refugee_data),]
crd$y_hat = predict.lm(ols, newdata=crd)
crd$y = crd$refugees
plot(y_hat~y, data=crd)

# $ python train.py --dset_id iiasa_unhcr_refugees_no_cat --task regression --epochs 500
# $ python sample.py --dset_id iiasa_unhcr_refugees_no_cat --task regression
saint = fread("~/git/saint/outputs/regression_iiasa_unhcr_refugees_no_cat.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))
refugee_data = fread("~/git/saint/data/iiasa_unhcr_refugees_no_cat.csv")
ols = lm(refugees~
           pop+
           gdp+
           urban, data=refugee_data
)
summary(ols)
crd = refugee_data[complete.cases(refugee_data),]
crd$y_hat = predict.lm(ols, newdata=crd)
crd$y = crd$refugees
plot(y_hat~y, data=crd)

# $ python train.py --dset_id iiasa_unhcr_refugees_long --task regression --pretrain
# $ python sample.py --dset_id iiasa_unhcr_refugees_long --task regression
saint = fread("~/git/saint/outputs/regression_iiasa_unhcr_refugees_long.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))
refugee_data = fread("~/git/saint/data/iiasa_unhcr_refugees_long.csv")
ols = lm(refugees~
           Region+
           year+
           variable, data=refugee_data
)
summary(ols)
crd = refugee_data[complete.cases(refugee_data),]
crd$y_hat = predict.lm(ols, newdata=crd)
crd$y = crd$refugees
plot(y_hat~y, data=crd)

# $ python train.py --dset_id iiasa_unhcr_displaced --task regression
# $ python sample.py --dset_id iiasa_unhcr_displaced --task regression
saint = fread("~/git/saint/outputs/regression_iiasa_unhcr_displaced.csv")
plot(saint)
summary(lm(y~y_hat, data=saint))
displaced_data = fread("~/git/saint/data/iiasa_unhcr_displaced.csv")
ols = lm(displaced_persons~
        displaced_persons_t1+
        displaced_persons_t2+
        displaced_persons_t3+
        mean_displaced_persons_t1_o1+
        mean_displaced_persons_t2_o1+
        mean_displaced_persons_t3_o1+
        mean_displaced_persons_t1_o2+
        mean_displaced_persons_t2_o2+
        mean_displaced_persons_t3_o2+
        mean_displaced_persons_t1_o3+
        mean_displaced_persons_t2_o3+
        mean_displaced_persons_t3_o3+
        Region+
        year+
        pop+
        gdp+
        urban, data=displaced_data
)
summary(ols)
crd = displaced_data[complete.cases(displaced_data),]
crd$y_hat = predict.lm(ols, newdata=crd)
crd$y = crd$displaced_persons
plot(y_hat~y, data=crd)


# $ python train.py --dset_id iiasa_unhcr_displaced2 --task regression
# $ python sample.py --dset_id iiasa_unhcr_displaced2 --task regression
saint = fread("~/git/saint/outputs/regression_iiasa_unhcr_displaced2.csv")
(sum(saint$y_hat)-sum(saint$y))/sum(saint$y)
plot(saint)
summary(lm(y~y_hat, data=saint))
displaced_data = fread("~/git/saint/data/iiasa_unhcr_displaced2.csv")
ols = lm(displaced_persons~
        pop_t1+
         pop_t2+
        pop_t3+
        gdp_t1+
        gdp_t2+
        gdp_t3+
        urban_t1+
        urban_t2+
        urban_t3+
        pop_t1_o1+
        gdp_t1_o1+
        urban_t1_o1+
        pop_t2_o1+
        gdp_t2_o1+
        urban_t2_o1+
        pop_t3_o1+
        gdp_t3_o1+
        urban_t3_o1+
        pop_t1_o2+
        gdp_t1_o2+
        urban_t1_o2+
        pop_t2_o2+
        gdp_t2_o2+
        urban_t2_o2+
        pop_t3_o2+
        gdp_t3_o2+
        urban_t3_o2+
        pop_t1_o3+
        gdp_t1_o3+
        urban_t1_o3+
        pop_t2_o3+
        gdp_t2_o3+
        urban_t2_o3+
        pop_t3_o3+
        gdp_t3_o3+
        urban_t3_o3+
        Region+
        year, data=displaced_data
)
summary(ols)
crd = displaced_data[complete.cases(displaced_data),]
crd$y_hat = predict.lm(ols, newdata=crd)
crd$y = crd$displaced_persons
plot(y_hat~y, data=crd)

# $ python train.py --dset_id iiasa_unhcr_displaced2_smol --task regression
# $ python sample.py --dset_id iiasa_unhcr_displaced2_smol --task regression
saint = fread("~/git/saint/outputs/regression_iiasa_unhcr_displaced2_smol.csv")
(sum(saint$y_hat)-sum(saint$y))/sum(saint$y)
plot(saint)
summary(lm(y~y_hat, data=saint))
displaced_data = fread("~/git/saint/data/iiasa_unhcr_displaced2_smol.csv")
ols = lm(displaced_persons~
           pop_t1+
           pop_t2+
           pop_t3+
           gdp_t1+
           gdp_t2+
           gdp_t3+
           urban_t1+
           urban_t2+
           urban_t3+
           Region+
           year, data=displaced_data
)
summary(ols)
crd = displaced_data[complete.cases(displaced_data),]
crd$y_hat = predict.lm(ols, newdata=crd)
crd$y = crd$displaced_persons
plot(y_hat~y, data=crd)


# $ python train.py --dset_id simple_uppsala_replication --task binary
# $ python sample.py --dset_id simple_uppsala_replication --task binary
saint = fread("~/git/saint/outputs/binary_simple_uppsala_replication.csv")
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
fit = glm(
  conflict~
    GDPcap+prec+tmax,
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
forecast$conflict[which(forecast$year>=2014)] = forecast$y_hat[which(forecast$year>=2014)]
forecast_agg = forecast[,.(
  conflicts=sum(y_hat, na.rm=T),
  GDPcap = mean(GDPcap, na.rm=T)
), by=.(scenario, year)]
library(scales)
ggplot(forecast_agg, aes(x=year,y=conflicts,group=scenario,color=scenario)) +
  scale_y_continuous(labels=label_comma()) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Global conflicts")


# $ python train.py --dset_id displacement_worldclim --task regression
# $ python sample.py --dset_id displacement_worldclim --task regression
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
ols = lm(
  displaced_persons~
    gdpgrowth+prec+tmax,
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
ols = lm(
  climate_disasters~
  area_sqkm+prec+tmax,
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


# $ python train.py --dset_id tripartite_bigram --task regression
# $ python sample.py --dset_id tripartite_bigram --task regression
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
forecast$humanitarian_needs[which(forecast$year > 2022)] = predict.lm(ols, newdata=forecast[which(forecast$year > 2022)])

forecast$humanitarian_needs = forecast$humanitarian_needs / 1e9
forecast_sub = subset(forecast, year > 2022 & year < 2101)
forecast_agg = data.table(forecast_sub)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario)]
library(scales)
ggplot(forecast_agg, aes(x=scenario,y=humanitarian_needs,fill=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_bar(stat="identity", position="dodge") +
  theme_classic() +
  labs(x="SSP Scenario", y="Humanitarian needs (billion USD$)",
       title="Projected global humanitarian needs (2023-2100)")
forecast_agg = data.table(forecast)[,.(
  humanitarian_needs=sum(humanitarian_needs, na.rm=T),
  displaced_persons=sum(displaced_persons, na.rm=T),
  climate_disasters=sum(climate_disasters, na.rm=T),
  conflict=sum(conflict, na.rm=T)
), by=.(scenario, year)]
ggplot(forecast_agg, aes(x=year,y=humanitarian_needs,group=scenario,color=scenario)) +
  scale_y_continuous(labels=dollar) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Humanitarian spend per donor (billion USD$)")
