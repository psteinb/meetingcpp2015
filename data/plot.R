#!/usr/bin/env Rscript

argv = commandArgs(TRUE)

if(length(argv) < 1){
    stop("Usage: plot.R <data_file(s)>")
}

top500_file <- argv[1]

output_base <- sub(".data","",top500_file)

                                        #top500_datasets <- read.table(top500_file, header = T,stringsAsFactors=TRUE)
library(stringr)
top500_datasets <- lapply(argv[1:length(argv)],
                          function(fn) {
                            loaded <- read.csv(fn, header = T, stringsAsFactors = T)
                            loaded$Top500_Year <- as.integer(str_extract(fn,"20[0-9]{2}"))
                            ds_id <- str_extract(fn,"20[0-9]{4}")
                            loaded$Top500_Month <- as.integer(substr(ds_id,5,7))

                            loaded
                          }
                          )

names(top500_datasets) <- argv[1:length(argv)]


library(dplyr)

merge.all <- function(x, y) {
    bind_rows(x, y)
  }

top500_reduced <- Reduce(merge.all, top500_datasets)

library(ggplot2)


my_theme <-  theme_bw() + theme(axis.title.x = element_text(size=20),
                                axis.title.y = element_text(size=20),
                                axis.text.x = element_text(size=16),
                                axis.text.y = element_text(size=16),
                                axis.text.x  = element_text()
                                ) 
my_theme <- my_theme + theme(legend.title = element_text(size=22, face="bold"),
                             legend.text = element_text( size = 20),
                             ## legend.text.align=0,
                             ## legend.title.align=0,
                             legend.position="top",
#                             legend.position=c(0, 1.),
                             legend.box.just="left",
                             
#                             legend.justification=c(0,0),
                             legend.key = element_rect(colour = 'white', fill = 'white', size = 0., linetype='dashed')) #+ theme(legend.title=element_blank()) 


                                        #head(top500_reduced)

items_2015 <- filter(top500_reduced,Top500_Year == 2015)

with_acc_2015 <- filter(items_2015,Accelerator.Co.Processor != factor("None"))

rank_limit <- 50
top25_reduced <- filter(top500_reduced,Rank <= rank_limit)

## pf <- filter(top25_reduced,Top500_Year > 2013, Rank<10)
## options(dplyr.width = Inf)
## select(pf, Name,Accelerator.Co.Processor,Accelerator,Top500_Year,Top500_Month,Rank, Computer)

refine <- function(input_df){
  input_df$merged_acc <- input_df$contained_co_proc
  input_df[is.na(input_df$merged_acc),]$merged_acc <- input_df[is.na(input_df$merged_acc),]$contained_acc
  input_df$acc_fraction <- input_df$merged_acc / input_df$occ
  input_df$date <- as.Date(paste(input_df$Top500_Year,input_df$Top500_Month,"1",sep="-"))
  input_df
}

top25 <- top25_reduced %>% group_by(Top500_Year, Top500_Month) %>% summarise(occ = n(), contained_co_proc = sum(Accelerator.Co.Processor != factor("None")), contained_acc = sum(!(Accelerator %in% factor("None"))))
top25 <- refine(top25)
top25$list <- factor(paste("Top",rank_limit,sep=""))
#top25

per_year <- top500_reduced %>% group_by(Top500_Year, Top500_Month) %>% summarise(occ = n(), contained_co_proc = sum(Accelerator.Co.Processor != factor("None")), contained_acc = sum(!(Accelerator %in% factor("None"))))
per_year <- refine(per_year)
per_year$list <- factor("Top500")
#per_year

compared <- rbind(per_year,top25)

## acc_plot <- ggplot(per_year, aes(x=date,y=acc_fraction))
## acc_plot <- acc_plot + geom_line() + ylim(0,1)
## acc_plot <- acc_plot + ylab("fraction of accelerator") + xlab("Year") 
## acc_plot <- acc_plot + my_theme #+ guides(col=guide_legend(ncol=3))

## ggsave("Top500_201x_acc_fraction.png",acc_plot)


## acc_plot <- ggplot(top25, aes(x=date,y=acc_fraction))
## acc_plot <- acc_plot + geom_line() + ylim(0,1)
## acc_plot <- acc_plot + ylab("fraction of accelerator") + xlab("Year") 
## acc_plot <- acc_plot + my_theme #+ guides(col=guide_legend(ncol=3))

## ggsave("Top25_201x_acc_fraction.png",acc_plot)


acc_plot <- ggplot(compared, aes(x=date,y=100*acc_fraction,color=list))
acc_plot <- acc_plot + geom_line(size=2) + ylim(0,50)
acc_plot <- acc_plot + ylab("accelerated locations / %") + xlab("Year") 
acc_plot <- acc_plot + my_theme #+ guides(col=guide_legend(ncol=3))

ggsave("201x_acc_fraction.png",acc_plot,width=8,height=6)
ggsave("201x_acc_fraction.svg",acc_plot,width=8,height=6)
