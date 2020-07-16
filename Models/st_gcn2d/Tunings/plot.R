library(ggplot2)
batch32 <- read.csv("2dADE13.csv",header = F)
batch32 <- t(batch32)
batch128 <- read.csv("2dADE15.csv",header = F)
batch128 <- t(batch128)
batch256 <- read.csv("2dADE14.csv",header = F)
batch256 <- t(batch256)

dat <- list(size32=batch32, size128=batch128, size256=batch256)
dat <- lapply(dat, function(x) cbind(x = seq_along(x), y = x))

list.names <- names(dat)
lns <- sapply(dat, nrow)
dat <- as.data.frame(do.call("rbind", dat))
dat$group <- rep(list.names, lns)
ggplot(dat, aes(x = x, y = V2, colour = group)) +
  theme_bw() +
  geom_line(linetype = "solid")


library(ggplot2)
drop0.3 <- read.csv("2dADE22(2).csv",header = F)
drop0.3 <- as.matrix(drop0.3)
drop0.5 <- read.csv("2dADE6(2).csv",header = F)
drop0.5 <- as.matrix(drop0.5)
drop0.7 <- read.csv("2dADE20.csv",header = F)
drop0.7 <- as.matrix(drop0.7)
drop0.55<- read.csv("2dADE21.csv", header = F)
drop0.55 <- as.matrix(drop0.55)

dat <- list(drop0.3=drop0.3, drop0.5=drop0.5, drop0.7=drop0.7, drop0.55=drop0.55)
dat <- lapply(dat, function(x) cbind(x = seq_along(x), y = x))

list.names <- names(dat)
lns <- sapply(dat, nrow)
dat <- as.data.frame(do.call("rbind", dat))
dat$group <- rep(list.names, lns)
ggplot(dat, aes(x = x, y = V1, colour = group)) +
  theme_bw() +
  geom_line(linetype = "solid")+labs(title="Drop Rate Tuning",
                                    x ="Epoch", y = "Loss")



library(ggplot2)

drop0.5 <- read.csv("2dADE6(2).csv",header = F)
drop0.5 <- as.matrix(drop0.5)
hidden512 <- read.csv("2dADE21(2).csv", header = F)
hidden512 <- as.matrix(hidden512)

dat <- list(drop0.5hidden256=drop0.5, drop0.5hidden512=hidden512)
dat <- lapply(dat, function(x) cbind(x = seq_along(x), y = x))

list.names <- names(dat)
lns <- sapply(dat, nrow)
dat <- as.data.frame(do.call("rbind", dat))
dat$group <- rep(list.names, lns)
ggplot(dat, aes(x = x, y = V1, colour = group)) +
  theme_bw() +
  geom_line(linetype = "solid")+labs(title="hidden layer size: 256 vs. 512",
                                     x ="Epoch", y = "Loss")



library(ggplot2)

drop0.3_1 <- read.csv("2dADE18.csv",header = F)
drop0.3_1 <- as.matrix(drop0.3_1)
drop0.3_2<- read.csv("2dADE22(2).csv", header = F)
drop0.3_2 <- as.matrix(drop0.3_2)


dat <- list(drop0.3_1=drop0.3_1, drop0.3_2=drop0.3_2)
dat <- lapply(dat, function(x) cbind(x = seq_along(x), y = x))

list.names <- names(dat)
lns <- sapply(dat, nrow)
dat <- as.data.frame(do.call("rbind", dat))
dat$group <- rep(list.names, lns)
ggplot(dat, aes(x = x, y = V1, colour = group)) +
  theme_bw() +
  geom_line(linetype = "solid")+labs(title="2 Trials of drop rate = 0.3",
                                     x ="Epoch", y = "Loss")

