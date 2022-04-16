library(stagePop)
solver.options=list(DDEsolver='PBS',tol=1e-8,hbsize=1e8,dt=0.01)

immigration <- function(mn,std,time) {
    theta <- std * std / mn
    k <- mn / theta
    return(dgamma(time,k,scale=theta))
}

solveDDE <- function(R1, R2) {
    init <- 100
    Tinit <- -40
    #
    ccFunctions <- list(
        deathFunc=function(stage,x,time,species,strain){return(0)},
        reproFunc=function(x,time,species,strain){return(0)},
        emigrationFunc=function(stage,x,time,species,strain){return(0)},
        develFunc=function(stage,x,time,species,strain){ # CONTINUOUS
            if (time > Tinit) {
                if (time < 20) {
                    if (stage == 1) {return(1/R1)} else {return(1e-13)}
                } else {
                    if (stage == 1) {return(1/R2)} else {return(1e-13)}
                }
            }
            return(1/R1)
        },
        durationFunc=function(stage,x,time,species,strain){ # INITIAL
            if (time == Tinit) {
                if (stage == 1) {return(R1)} else {return(1e13)}
            }
        },
        immigrationFunc=function(stage,x,time,species,strain){
            if (stage==1 && time<=0) {return(v <- immigration(40,5,time-Tinit+20))}
            return(0)
        }
    )
    #
    out <- popModel(
        numSpecies=1,
        numStages=2,
        timeDependLoss=FALSE,
        timeDependDuration=TRUE,
        ICs=list(matrix(0,nrow=2,ncol=1)),
        timeVec = seq(Tinit,50,0.01),
        solverOptions=solver.options,
        rateFunctions=ccFunctions,
        stageNames=list(c('juveniles','adults')),
        speciesNames=c('Culex'),
        saveFig = FALSE,
        plotFigs = FALSE
    )
    #
    return(data.frame("time"=out[,1],
                      "size"=out[,2]))
}

outS <- solveDDE(40,40) # 20+-5
outM <- solveDDE(60,40)
outL <- solveDDE(60,60) # 40+-5

out <- data.frame("time"=outM$time,
                  "sizeS"=outS$size,
                  "sizeM"=outM$size,
                  "sizeL"=outL$size)

write.csv(out,"mat/solveDDE_out.csv",row.names=FALSE)
