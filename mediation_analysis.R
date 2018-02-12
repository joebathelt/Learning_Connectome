library(lavaan)
setwd('U:/joe_python/GitHub/Learning_Connectome')
data <- read.csv('./data/hub_mediation.csv')
data[colnames(data)[2:length(colnames(data))]] <- scale(data[colnames(data)[2:length(colnames(data))]])
data['Age'] = data['Age']**2

# Reading and clustering coefficient
model <- ' # direct effect
Reading ~ c*Age
# mediator
hub_clustering ~ a*Age
Reading ~ b*hub_clustering
# indirect effect (a*b)
ab := a*b
# total effect
total := c + (a*b)
'
fit <- sem(model, data = data)
summary(fit, fit.measures=TRUE)

# Reading and global efficiency
model <- ' # direct effect
Reading ~ c*Age
# mediator
hub_efficiency ~ a*Age
Reading ~ b*hub_efficiency
# indirect effect (a*b)
ab := a*b
# total effect
total := c + (a*b)
'

fit <- sem(model, data = data)
summary(fit, fit.measures=TRUE)

# Maths and clustering coefficient
model <- ' # direct effect
Maths ~ c*Age
# mediator
hub_clustering ~ a*Age
Maths ~ b*hub_clustering
# indirect effect (a*b)
ab := a*b
# total effect
total := c + (a*b)
'

fit <- sem(model, data = data)
summary(fit, fit.measures=TRUE)

# Maths and global efficiency
model <- ' # direct effect
Maths ~ c*Age
# mediator
hub_efficiency ~ a*Age
Maths ~ b*hub_efficiency
# indirect effect (a*b)
ab := a*b
# total effect
total := c + (a*b)
'

fit <- sem(model, data = data)
summary(fit, fit.measures=TRUE)
