## Load dependencies 
pacman::p_load(tidyverse,gridExtra,treemapify,ggfittext)
ggplot2::theme_set(theme_minimal())

## TRAINING TIME 
model <- c("CNN","BERT-CNN","BERT-GRU")
major <- c(9*60+19,36*60+17,42*60+44)
minor <- c(11*60+12,34*60+45,80*60+20)

msdf <- data.frame(cbind(model,major,minor))

msdf$model <- factor(msdf$model, levels=c("CNN","BERT-CNN","BERT-GRU"))

numcols <- c("major", "minor")

msdf[numcols] <- sapply(msdf[numcols],as.character)
msdf[numcols] <- sapply(msdf[numcols],as.numeric)

msdf %>% 
  pivot_longer(cols = c("major","minor"),
               names_to = "category",
               values_to = "seconds") %>% 
  ggplot(aes(x=model,y=seconds,fill=category)) + 
  geom_col(position="dodge") + 
  labs(x="",fill="") + 
  # scale_fill_grey() +
  scale_fill_viridis_d(option = "C", begin = 0.1, end = 0.6) +
  theme(legend.position="top",
        legend.justification="right",
        legend.text = element_text(size = rel(1.2)),
        axis.text.x = element_text(size= rel(1.5)) 
  )

ggsave("figures/training_time.png", width = 20, height = 10, units = "cm")

## OVERFITTING :: ACCURACY AND LOSS FOR CNN MAJOR 

df <- read_csv("data/e02_cnnmajor_results.csv")

p.overfit1 <- df %>% 
  ggplot(aes(x=ep)) + 
  geom_line(aes(y=tloss),color="#5DC863") + 
  geom_line(aes(y=vloss),color="#3B528B")

p.overfit2 <- df %>% 
  ggplot(aes(x=ep)) + 
  geom_line(aes(y=tacc),color="#5DC863") + 
  geom_line(aes(y=vacc),color="#3B528B")

df %>% 
  pivot_longer(cols=colnames(df[2:5]),
               names_to="metric",
               values_to="n") %>% 
  mutate(
    eval_metric = case_when(
      metric=="tacc" ~ "Accuracy",
      metric=="vacc" ~ "Accuracy",
      metric=="tloss"~ "Loss",
      metric=="vloss" ~ "Loss"
    ),
    split = case_when(
      metric=="tacc" ~ "Training",
      metric=="vacc" ~ "Validation",
      metric=="tloss"~ "Training",
      metric=="vloss" ~ "Validation"
    )) %>% 
  ggplot(aes(x=ep,y=n,color=split)) + 
  facet_wrap(~eval_metric,scales="free") + 
  geom_line() +
  labs(x="# Epochs",y="",color="Data") +
  # scale_color_grey(start=0.3,end=0.7) + 
  scale_color_viridis_d(option = "C", begin = 0, end = 0.75) +
  theme(strip.text = element_text(size = rel(1.2)), 
        legend.position = "bottom",
        legend.justification = "right")

ggsave("figures/cnnmaj_acc_loss.png", width = 20, height = 10, units = "cm")

## BERTCNN RESULTS - NO OVERFITTING
df <- read_csv("data/BERTCNN_minor_results.csv")

x <- rep(df$ep,4)
y <- c(df$tacc,df$vacc,df$tloss,df$vloss)
z <- c(rep("Training Accuracy",10),rep("Validation Accuracy",10),rep("Training Loss",10),rep("Validation Loss",10))

df1 <- as.data.frame(cbind(x,y,z))
df1$y <- y

df1$x <- as.numeric(as.character(df1$x))

### REDUX
df1 %>% 
  mutate(
    z1 = case_when(
      z=="Training Accuracy" ~ "Accuracy",
      z=="Training Loss" ~ "Loss",
      z=="Validation Accuracy" ~ "Accuracy",
      z=="Validation Loss" ~ "Loss"
      ),
    z2 = case_when(
    z=="Training Accuracy" ~ "Training",
    z=="Training Loss" ~ "Training",
    z=="Validation Accuracy" ~ "Validation",
    z=="Validation Loss" ~ "Validation"
    )) %>% 
  ggplot(aes(x=x,y=y,color=z2)) + 
  facet_wrap(~z1,scales="free") + 
  geom_line() +
  labs(x="# Epochs",y="",color="Data") +
  scale_x_discrete(limits=seq(1,10,by=1)) +
  # scale_color_grey(start=0.3,end=0.7) + 
  scale_color_viridis_d(option = "C", begin = 0, end = 0.75) +
  theme(strip.text = element_text(size = rel(1.2)),
        legend.position = "bottom",
        legend.justification = "right")

ggsave("figures/bertcnnmin_acc_loss.png", width = 20, height = 10, units = "cm")

## HEATMAP 

cnnbert_results <- read_csv("data/bertcnn-cls-report.csv") %>% 
  rename(Precision = precision,
         Recall = recall,
         `F1 Score` = `f1-score`)

metrics <- c("Precision","Recall","F1 Score")

long.df <- tidyr::pivot_longer(cnnbert_results,
                               metrics,
                               names_to = "metric",
                               values_to = "n") %>%
  dplyr::mutate(
    metric = factor(metric, levels = c("Precision", "Recall", "F1 Score")),
    Major = major
  )

# classic ggplot, with text in aes
ggplot(long.df, aes(Major, fct_rev(metric), fill=n)) + 
  geom_tile(color="white") + 
  scale_x_discrete(limits=c(0:7)) + 
  scale_fill_viridis_c(option="plasma",begin=0,end=.9) +
  labs(y="") + 
  coord_equal() +
  theme(legend.title=element_blank(),
        axis.title = element_text(size = rel(1.5)),
        axis.text = element_text(size = rel(1.4))) 

ggsave("figures/prf1-major.png", width = 20, height = 10, units = "cm")

