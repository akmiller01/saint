list.of.packages <- c("data.table","OpenML", "farff", "glm")
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

# $ python train.py --dset_id iiasa_unhcr_refugees --task regression
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
