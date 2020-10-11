library(ggplot2)
setwd("Documents/Master/MasterPraktikum/pycharm_folder/prediction-of-binding-affinity/building_model/davis_folder/")

{
setwd("../davis_folder/")
data <- read.csv("pkd_cleaned_interactions.csv")
data$cids <- NULL
data <- data.frame(unlist(data))
sum(data$unlist.data. > 7)/length(data$unlist.data.)

qplot(data$unlist.data., geom="histogram", bins=30) + 
  theme_bw() + 
  geom_vline(xintercept = 7, color = "red") +
  ylab(label = "Number of DT pairs") +
  xlab(label = "pKd values") +
  ggtitle("Davis (6.8% are labeled as binding)")
ggsave("davis_pKd_frequencies.png")
}#Davis


{
setwd("../kiba_folder/")

data <- read.csv("cleaned_interactions.csv")
data$cids <- NULL
data <- data.frame(unlist(data))
data <- na.omit(data)
sum(data$unlist.data. > 12.1)/length(data$unlist.data.)

qplot(data$unlist.data., geom="histogram", bins=30) + 
  theme_bw() + 
  geom_vline(xintercept = 12.1, color = "red") +
  ylab(label = "Number of DT pairs") +
  xlab(label = "Kiba Scores") +
  ggtitle("Kiba (17.2% are labeled as binding)")
ggsave("kiba_values_frequencies.png")
}#Kiba


{
setwd("../bdb_pkd_folder/")

data <- read.csv("pkd_cleaned_interactions.csv")
data$cids <- NULL
data <- data.frame(unlist(data))  
data <- na.omit(data)
sum(data$unlist.data. > 7)/length(data$unlist.data.)

qplot(data$unlist.data., geom="histogram", bins=30) + 
  theme_bw() + 
  geom_vline(xintercept = 7, color = "red") +
  ylab(label = "Number of DT pairs") +
  xlab(label = "pKd values") +
  ggtitle("BindingDB pKd (18.3% are labeled as binding)")
ggsave("BindingDB_pKd_frequencies.png")
}#bindingDB