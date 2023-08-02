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

# $ python train.py --dset_id conflict_clim --task binary
# $ python sample.py --dset_id conflict_clim --task binary
saint = fread("~/git/saint/outputs/binary_conflict_clim.csv")
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
conflict_clim = fread("~/git/saint/data/conflict_clim.csv")
fit = glm(
  conflict~
    prec+tmin+tmax+iso3,
  data=conflict_clim, family="binomial"
)
summary(fit)
nullhypo <- glm(conflict~1, data=conflict_clim, family="binomial")
mcFadden = 1-logLik(fit)/logLik(nullhypo)
mcFadden
forecast = fread("~/git/saint/data/conflict_clim_forecasting.csv")
forecast$y_prob = predict.glm(fit, newdata = forecast)
forecast$y_prob = forecast$y_prob - min(forecast$y_prob, na.rm=T)
forecast$y_prob = forecast$y_prob / max(forecast$y_prob, na.rm=T)
forecast$y_hat = round(forecast$y_prob)
forecast$conflict[which(forecast$year>=2014)] = forecast$y_hat[which(forecast$year>=2043)]
forecast_agg = forecast[,.(
  conflicts=sum(y_hat, na.rm=T)
), by=.(scenario, year)]

ggplot(forecast_agg, aes(x=year,y=conflicts,group=scenario,color=scenario)) +
  scale_y_continuous(labels=label_comma(), limits = c(0, max(forecast_agg$conflicts))) +
  geom_line() +
  theme_classic() +
  labs(x="Year", y="Global conflicts")
