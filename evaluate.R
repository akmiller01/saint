list.of.packages <- c("data.table","OpenML", "farff")
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

# $ python train.py --dset_id ssp1_ext --task multiclass
# $ python sample.py --dset_id ssp1_ext --task multiclass
saint = fread("~/git/saint/outputs/multiclass_ssp1_ext.csv")
boxplot(saint)
pred = data.table(table(saint$y, saint$y_hat))
names(pred) = c("y", "y_hat", "count")
pred$correct = pred$y == pred$y_hat
