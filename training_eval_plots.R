library(ggplot2)

## read in training log file
df_raw <- read.table(pipe("awk 'BEGIN{i=0}{i++;if (i%2==1) print }' < training_log.txt"), sep=";")

## rename and split columns from training log 
column_names <- c("episode", "steps", "time", "epsilon", "tot_reward", "final_pos", "avg_reward")
columnSplitter <- function(i){
  split_column <- reshape2::colsplit(df_raw[,i], pattern=": ", names=c("", column_names[i]))[,column_names[i]]
  out_df <- data.frame(split_column)
  return(out_df)
}
column_list <- lapply(1:length(column_names), FUN=columnSplitter)
df <- do.call(cbind, column_list)
names(df) <- column_names

## calculate rolling avg
df$avg_tot_reward <- zoo::rollapply(df$tot_reward, width = 100, FUN = mean, align = "right", partial=T)
df$avg_final_pos <- zoo::rollapply(df$final_pos, width = 100, FUN = mean, align = "right", partial=T)
df$wins_100_window <- zoo::rollapply(ifelse(df$final_pos==3161,1,0), width = 100, FUN = sum, align = "right", partial=T)
df$win_perc <- df$wins_100_window/100

## plot 
reward_plot <- ggplot(data=df, aes(x=episode)) + 
  geom_line(aes(y=tot_reward), alpha=0.15) +
  geom_line(aes(y=avg_tot_reward), alpha=0.75) +
  # geom_line(aes(y=avg_final_pos), alpha=0.75, color="orange") +
  theme_classic() + 
  labs(x="Episode", y="Reward")

ggsave("reward_plot.png", width=4, height=2.5, units=c("in"))

win_plot <- ggplot(data=df, aes(x=episode)) + 
  geom_line(aes(y=win_perc), alpha=0.75) +
  theme_classic() + 
  labs(x="Episode", y="Win % (avg. 100 episodes)")

ggsave("win_plot.png", width=4, height=2.5, units=c("in"))