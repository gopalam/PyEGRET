# Regenerate the EGRETci reference results in tests/reference/egretci/.
# Run from this folder:  Rscript make_references.R
# Takes about an hour and a half, most of it in runGroupsBoot.
# The .rds files it writes were converted to the JSON files beside them with
# Python (rdata), keeping only what the tests compare.
source("load_sources.R")
eList <- Choptank_eList
load(file.path(ref, "..", "tests", "reference", "egretci", "r", "Choptank_caseSetUp.RData"))

# Small pieces: block samples, the Bayesian interval, pVal, wordLike.
s <- eList$Sample
b1 <- blockSample(s, 200, startSeed = 494818)
b2 <- blockSample(s, 150, startSeed = 7)
cases <- data.frame(x = c(0, 3, 10, 20, 39), n = c(10, 10, 20, 20, 39))
bb <- do.call(rbind, lapply(seq_len(nrow(cases)), function(i)
  binom::binom.bayes(cases$x[i], cases$n[i], 0.7, "central")[, c("x", "n", "mean", "lower", "upper")]))
vals <- c(-0.3, 0.2, 0.5, 0.1, -0.05, 0.7, 0.9, 0, NA, 0.4)
pv <- c(pVal(vals), pVal(abs(vals)), pVal(c(1, 2, -1, 3, 4, 5, -2, 6)))
saveRDS(list(julian1 = b1$Julian, conc1 = b1$ConcHigh, julian2 = b2$Julian, binom = bb,
             pval = pv, words = wordLike(c(0.02, 0.5, 0.92, 0.999))), "out_small.rds")

# runPairsBoot: the EGRETci vignette case, and the plain one.
pr2 <- runPairs(eList, year1 = 1985, year2 = 2010, windowSide = 7, flowBreak = TRUE,
                Q1EndDate = "1995-05-31", wall = TRUE, sample1EndDate = "1995-05-31",
                QStartDate = "1979-10-01", QEndDate = "2010-09-30", paStart = 4, paLong = 5,
                verbose = FALSE)
saveRDS(list(pairResults = pr2, boot = runPairsBoot(eList, pr2, nBoot = 100)),
        "out_pairs_wall_season.rds")
pr <- runPairs(eList, year1 = 1985, year2 = 2010, windowSide = 7, verbose = FALSE)
saveRDS(list(pairResults = pr, boot = runPairsBoot(eList, pr, nBoot = 30)), "out_pairs_simple.rds")

# wBT with EGRETci's example set-up.
saveRDS(wBT(eList, Choptank_caseSetUp, saveOutput = FALSE), "out_wbt.rds")

# Confidence bands: the bootstrap annual replicates and ciBands.
eListM <- suppressMessages(modelEstimation(eList, verbose = FALSE))
reps <- lapply(1:5, function(n) bootAnnual(eListM, 200, 494817 + n))
saveRDS(list(reps = reps, ci = ciBands(eListM, reps, c(0.05, 0.95))), "out_ci_model.rds")
eListS <- runSeries(eList, windowSide = 7, verbose = FALSE)
repsS <- lapply(1:3, function(n) bootAnnual(eListS, 200, 494817 + n))
saveRDS(list(reps = repsS, ci = ciBands(eListS, repsS, c(0.05, 0.95))), "out_ci_series.rds")

# genDailyBoot: two bootstrap replicates with the same Kalman seed.
eListK <- WRTDSKalman(eListM, niter = 5, seed = 1, verbose = FALSE)
gdb <- suppressMessages(genDailyBoot(eListK, nBoot = 2, nKalman = 1, setSeed = 1))
cat("genDailyBoot replicates identical:", identical(gdb[, 1], gdb[, 2]), "\n")

# runGroupsBoot with a wall (slow).
g <- runGroups(eList, group1firstYear = 1995, group1lastYear = 2004,
               group2firstYear = 2005, group2lastYear = 2010, windowSide = 7, wall = TRUE,
               sample1EndDate = "2004-10-30", paStart = 4, paLong = 2, verbose = FALSE)
saveRDS(list(groupResults = g, boot = runGroupsBoot(eList, g, nBoot = 20)), "out_groups_wall.rds")
