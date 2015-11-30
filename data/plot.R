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


my_theme <- my_theme + theme(line = element_line(colour = "black", size = 0.5, linetype = 1,lineend = "butt"),
                             rect = element_rect(fill = "white", colour = "black", size = 0.5, linetype = 1),
                             ## text = element_text(#family = base_family, face = "plain",
                             ##   colour = "black", size = 12#,
                             ##   #hjust = 0.5, vjust = 0.5, angle = 0, lineheight = 0.9
                             ##   ),
                             axis.text =  element_text(size = rel(0.8), colour = "white"),
                             strip.text = element_text(size = rel(0.8), colour = "white"),

                             axis.line =          element_blank(),
                             axis.text.x =        element_text(vjust = 1),
                             axis.text.y =        element_text(hjust = 1),
                             axis.ticks =         element_line(colour = "white", size = 0.2),
                             axis.title =         element_text(colour = "white"),
                             axis.title.x =       element_text(vjust = .2),
                             axis.title.y =       element_text(angle = 90),
                                        #    axis.ticks.length =  unit(0.3, "lines"),
                             ## axis.ticks.margin =  unit(0.5, "lines"),
                             
                             legend.background =  element_rect(colour = NA, fill="black",size=0),
                             ## legend.margin =      unit(0.2, "cm"),
                             legend.key =         element_rect(fill = "black", colour = "black"),
                             ## legend.key.size =    unit(1.2, "lines"),
                             legend.key.height =  NULL,
                             legend.key.width =   NULL,
                             legend.text =        element_text(size = rel(0.8), colour = "white"),
                             legend.text.align =  NULL,
                             legend.title =       element_text(size = rel(0.8), face = "bold", hjust = 0, colour = "black"),
                             legend.title.align = NULL,
                             legend.position =    "right",
                             legend.direction =   "horizontal",
                             legend.justification = "center",
                             
                             panel.background =   element_rect(fill = "black", colour = NA),
                             panel.border =       element_rect(fill = NA, colour = "white"),
                             panel.grid.major =   element_line(colour = "grey20", size = 0.2),
                             panel.grid.minor =   element_line(colour = "grey5", size = 0.5),
                             ## panel.margin =       unit(0.25, "lines"),
                             
                             strip.background =   element_rect(fill = "grey30", colour = "grey10"),
                             strip.text.x =       element_text(),
                             strip.text.y =       element_text(angle = -90),
                             
                             plot.background =    element_rect(colour = "black", fill = "black"),
                             plot.title =         element_text(size = rel(1.2)),
                             ## plot.margin =        unit(c(1, 1, 0.5, 0.5), "lines"),
                             
                             complete = TRUE)

my_theme <- my_theme + theme(legend.text = element_text( size = 20),
                             legend.position="top",
                             legend.box.just="left",
                             legend.key = element_rect(colour = 'white', fill = 'black', size = 0., linetype='dashed'),
                             legend.background =  element_rect(colour = "white",fill="black"))


items_2015 <- filter(top500_reduced,Top500_Year == 2015)

with_acc_2015 <- filter(items_2015,Accelerator.Co.Processor != factor("None"))

rank_limit <- 50
top25_reduced <- filter(top500_reduced,Rank <= rank_limit)

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

acc_plot <- ggplot(compared, aes(x=date,y=100*acc_fraction,color=list))
acc_plot <- acc_plot + geom_line(size=2) + ylim(0,50)
acc_plot <- acc_plot + ylab("accelerated locations / %") + xlab("Year") 
acc_plot <- acc_plot + my_theme #+ guides(col=guide_legend(ncol=3))
acc_plot <- acc_plot + labs(fill="",color="")
ggsave("201x_acc_fraction.png",acc_plot,width=8,height=5)
ggsave("201x_acc_fraction.svg",acc_plot,width=8,height=5)
