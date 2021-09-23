library(dplyr)
library(randomForest)
library(knitr)
library(haven)
library(tidyverse)
library(plyr)

nepal_flat_full_i <- read_sav("/home/yonghyun/Dropbox/United Nations/Use_cases/Nepal_use_case/OriginalData/2011 DHS/NPIR60FL.SAV")
non_vars <- nepal_flat_full_i %>% names %>% str_detect("\\$") %>% which
nepal_i <- nepal_flat_full_i %>% select(-all_of(non_vars)) 

nepal_i <- distinct(nepal_i, cbind(V001, V002), .keep_all = TRUE)
#nrow(unique(cbind(nepal_i$V001, nepal_i$V002)))

#age
table(nepal_i$V012)

#Region
tmpmat <- model.matrix(~factor(V101), nepal_i)[, -1]
nepal_i$V101_Hill <- tmpmat[,1]
nepal_i$V101_Terai <- tmpmat[,2]
all((nepal_i$V101 == 2) == nepal_i$V101_Hill)
all((nepal_i$V101 == 3) == nepal_i$V101_Terai)
#nepal_i$V101 <- labelled(nepal_i$V101 - 1, c(Mountain = 0, Hill = 1, Terai = 2), label = "Region")
#table(nepal_i$V101)
#unique(nepal_i$V101)

#ResidenceType
nepal_i$V102_Rural <-model.matrix(~factor(V102), nepal_i)[, -1]
all((nepal_i$V102 == 2) == nepal_i$V102_Rural)
#nepal_i$V102 <- labelled(nepal_i$V102 - 1, c(Urban = 0, Rural = 1), label = "Type of place of residence")
#table(nepal_i$V102)
#unique(nepal_i$V102)

#Religion
nepal_i <- nepal_i[nepal_i$V130 != 96,]
tmpmat <- model.matrix(~factor(V130), nepal_i)[, -1]

nepal_i$V130_Buddhist <- tmpmat[,unique(nepal_i$V130)[1] - 1]
nepal_i$V130_Muslim <- tmpmat[,unique(nepal_i$V130)[2] - 1]
nepal_i$V130_Kirat <- tmpmat[,unique(nepal_i$V130)[3] - 1]
nepal_i$V130_Christian <- tmpmat[,unique(nepal_i$V130)[4] - 1]

all((nepal_i$V130 == 2) == nepal_i$V130_Buddhist)
all((nepal_i$V130 == 3) == nepal_i$V130_Muslim)
all((nepal_i$V130 == 4) == nepal_i$V130_Kirat)
all((nepal_i$V130 == 5) == nepal_i$V130_Christian)

#Ethnicity

#Education in single years
unique(nepal_i$V133)
table(nepal_i$V133)

# number of children
unique(nepal_i$V137)
table(nepal_i$V137)

# number of women
unique(nepal_i$V138)
table(nepal_i$V138)

# Sex of the head of the household
nepal_i$V151_Female <-model.matrix(~factor(V151), nepal_i)[, -1]
all((nepal_i$V151 == 2) == nepal_i$V151_Female)
unique(nepal_i$V151)
table(nepal_i$V151)

#Random variable Y
unique(nepal_i$V313)
nepal_i$V313_yes <- ifelse(nepal_i$V313 == 0, 0, 1)

#########################################
head(model.matrix(~factor(V101), nepal_i))[,-1]

unique(cbind(nepal_i$V101, nepal_i$V102))
unique(cbind(nepal_i$V101, nepal_i$V131))
ddply(nepal_i, .(nepal_i$V101, nepal_i$V131), nrow)
unique(nepal_i$V101)
unique(nepal_i$V102)
table(nepal_i$V102)
table(nepal_i$V101)
sum(nepal_i$V101 == 1 & nepal_i$V102 == 1)
sum(nepal_i$V101 == 3 & nepal_i$V102 == 2)

unique(nepal_i$V104)
unique(nepal_i$V105)
unique(nepal_i$V130)
table(nepal_i$V130)
unique(nepal_i$V131)
table(nepal_i$V131)

unique(nepal_i$V133)
table(nepal_i$V133)

unique(nepal_i$V137)
table(nepal_i$V137)

table(nepal_i$V138)

unique(nepal_i$V151)
table(nepal_i$V151)

table(nepal_i$V313)
unique(nepal_i$V313)

"V013R","V113", "V128", "V116", "V129", "V106"
unique(nepal_i$V218R)

unique(nepal_i$V150) ##

#unique(nepal_i$V013R)

unique(nepal_i$V113) ##
table(nepal_i$V113)

unique(nepal_i$V128) ##
table(nepal_i$V128)

unique(nepal_i$V116) ##
table(nepal_i$V116)

unique(nepal_i$V129) ##
table(nepal_i$V129)

unique(nepal_i$V106) 

V013R
unique(nepal_i$V313)
####################################################

write.csv(nepal_i, "/home/yonghyun/Documents/GLMM/RealData/nepal_i.csv")
getwd()
