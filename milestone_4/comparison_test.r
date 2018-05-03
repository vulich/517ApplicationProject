library(lme4)
library(nlme)
library(multcomp)
data = scan("errors.txt")
errors = matrix(data,byrow=T,ncol=4)
Model = factor(errors[,1])
PCA = factor(errors[,2])
Split = factor(errors[,3])
Error = errors[,4]
summary(lme(Error~Model*PCA,random=~1|Split))
summary(glht(lmer(Error~Model*PCA+(1|Split)),linfct=mcp(Model="Tukey")))
summary(glht(lmer(Error~Model*PCA+(1|Split)),linfct=mcp(PCA="Tukey")))
