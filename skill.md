---
name: overgo-iteration
description: The rules for Overgo development, written as one logical statement per line. Use for implementation and explicit continue, resume or loop requests. Refactor as needed to optimize development pace, reduce code and test surface area and improve processing times and memory usage based on observed behavior.
---

# Intent

- overgo = one Go-native system that serves, trains, evaluates and composes models on consumer hardware (check: internal/repoanalysis TestRSIRuntimeIsGoOnly, internal/repoanalysis TestPythonLivesOnlyInItsIgnoredPlace)
- purpose = recursive self-improvement through its own inference and training, unattended
- goal = improvement in what the system serves = the scorecard (`go run ./cmd/validate -scorecard`)
- scorecard columns = `first-output`, `throughput`, `held-out-quality`, `failure-rate`, `load-time`, `peak-device-memory`, `setup-steps` (check: cmd/validate TestScorecardRowsEveryServableModel, cmd/validate TestScorecardHasNoUnexplainedCell)
- loop = measure -> find the gap -> train or change -> judge on held-out data
- row filed <=> it closes the loop or speeds the loop or serves inference or serves training
- progress = each scorecard cell compared with its previous measurement (check: internal/scorecard TestEarlierIsEachCellsNewestEarlierMeasurement, cmd/loop TestCycleComparesWithTheNewestEarlierScorecardWithoutAMain, cmd/validate TestScheduledScorecardPublishesDeltas)
- never a "passed" state and never a promotion (check: internal/repoanalysis TestRootDocumentsNameOnlyWhatTheTreeHolds, internal/plan TestLivePlanRejectsRetainedCompletionState)
- `go run ./cmd/loop -improve` = measure -> publish the scorecard -> file each regression as a row (check: cmd/loop TestTheGapsAreFiledOnceTheScorecardPublishes, internal/loop TestCycleFilesDeclinesAndAcceptsNothing, cmd/loop TestScorecardGapsBecomeProposals)
- `master` takes landings (check: internal/landing TestWorktreeCandidateIsIntegratedOntoTheTrunk, internal/landing TestTrunkCheckoutFollowsWithItsOwnChangesKept)
- local `main` = GitHub staging, which nothing reads (check: cmd/loop TestCycleComparesWithTheNewestEarlierScorecardWithoutAMain, cmd/loop TestPushPushesTheNamedCommitOnlyWhenAsked)
- push <=> a request to push, never the system's own initiative (check: cmd/loop TestPushPushesTheNamedCommitOnlyWhenAsked)
- this file and the plan's standing paragraphs = the system's doctrine
- doctrine = how the system works, which every session and the loop follow and correct on evidence
- doctrine != any person's preferences or rules (check: cmd/loophook TestDoctrineNamesNoPerson, internal/plan TestControlVocabularyNamesNoPerson)
- a rule that evidence shows costly or wrong => correct it in the doctrine != cite it as authority
- requests and models propose (check: internal/runrecord TestProposerEvaluatorDeciderAuthorityDomainsAndPriorGeneration)
- goals, budgets and stops = inputs the system follows (the plan's intent, resource ceilings, a recorded stop) and never sets itself (check: cmd/plan TestEnforceStopReason, internal/landing TestChangeIsJudgedByTheTermsItStartedWith, cmd/loop TestPersistentLoopStop)
- executable policy admits and activates (check: internal/agenttool TestCandidateCatalogStagesVerifiesAndActivatesExactly, internal/runrecord TestProposerEvaluatorDeciderAuthorityDomainsAndPriorGeneration)
- OvergoDB holds state
- the plan holds implementation (check: internal/plan TestSinglePlanAuthority)
- every change aims at capability or performance or development velocity or memory use
- bold change > cautious increment (a rewritten slow path, a deleted subsystem, a new kernel, a different data layout)
- framework = the gate + the held-out scorecard + the cycle
- the framework catches regressions and broken code (check: cmd/gate TestSmallGateLandsAChange, internal/loop TestCycleFilesDeclinesAndAcceptsNothing)
- bold change => write the test that would have caught its failure -> let the framework judge
- never shrink a change to what can be checked by hand
- never hedge a change with flags or shims
- never defer a change for fear of breaking what the framework checks
- refused landing = information != failure (check: internal/landing TestRecoveryDrills)
- ambition governs the code and never the system's boundaries (check: internal/landing TestChangeIsJudgedByTheTermsItStartedWith, cmd/loop TestPushPushesTheNamedCommitOnlyWhenAsked)
- boundaries = scope, budgets, stops, pushes and downloads, which the system follows and never changes (check: cmd/loop TestPushPushesTheNamedCommitOnlyWhenAsked, cmd/plan TestEnforceStopReason, internal/landing TestChangeIsJudgedByTheTermsItStartedWith)
- keep correctness and quality
- simpler implementation > larger one (check: internal/landing TestItemsGrowthIsChargedAcrossItsLandings, internal/landing TestWrittenCodeIsCountedByKind)
- ambition = size of the gain != size of the change

## Terms

A term used in this file is plain English or is defined here.

- lead = the session that holds the main checkout, ranks the plan and keeps the doctrine current
- session = one running agent
- row = one unit of work in `docs/plan.json`
- step = one landing's worth of a row, with its check (`verify`) (check: internal/plan TestOpenStepRequiresVerifier, internal/landing TestLandingCompletesSeveralSteps)
- gate = the automated checks a commit must pass
- landing = a commit made through the gate
- lane = one of the gate's longer test runs (device, browser, cuda)
- cycle = the full measurement run (`go run ./cmd/loop -improve`)
- cell = one model, task and measure on the scorecard
- home = the one package that holds a mechanic
- leftover = a file that a run, a build or a session wrote and that no later work reads
- place = a directory where leftovers accumulate

A line of this file reads by this notation, and cites the tests that hold its rule as `(check: <package> <test>, ...)`.

| Write | Read as |
| --- | --- |
| `A => B` | if A then B |
| `A <=> B` | A exactly when B |
| `A -> B` | A then B, or A goes to B |
| `A = B` | A is B |
| `A != B` | A is not B |
| `A > B` | prefer A over B |
| `never A` | A is forbidden |
| `A and B`, `A or B` | both, either |
| `A because B` | B is the reason for A |
| `A except B` | A holds unless B |

## Writing

- a rule = code => it cites the tests that fail when it is broken, or it goes (check: internal/landing TestEveryRuleCitesItsCheck)
- a semicolon never joins two clauses of a rule, because it hides their relation (check: internal/landing TestTheDoctrineHoldsItsOwnForm)
- history (a date, a commit id) -> the commit message, never the doctrine (check: internal/landing TestTheDoctrineHoldsItsOwnForm)
- each rule has one line (check: internal/landing TestTheDoctrineHoldsItsOwnForm)
- doc comment's first line = `Name <verb> ...`, one plain sentence (check: internal/repoanalysis TestGoStyleCensus)

## Fewer components

- net production lines flat or falling except the row declares growth (check: internal/landing TestItemsGrowthIsChargedAcrossItsLandings, internal/landing TestChangeIsJudgedByTheTermsItStartedWith, internal/landing TestWrittenCodeIsCountedByKind)
- growth that buys capability or speed or memory = declared and welcome (check: internal/landing TestLandingRaisesItsOwnCount, cmd/plan TestSetCountWritesTheCountAndItsReason)
- never a regex over structured data (check: internal/repoanalysis TestNoRegexParsesStructuredData)
- path only tests reach = unwired => add its route or lane or command (check: internal/repoanalysis TestEveryProcessDeclarationIsReachedByACommand, internal/repoanalysis TestEveryProcessRecordSchemaHasAProductionWriter, internal/landing TestExportUseCountsConsumers)
- never markdown behavior control => an idea becomes a Go test or a gate rule (check: internal/repoanalysis TestRSIRuntimeIsGoOnly)

## Autonomous feedback

- stay inside the scope and resource ceilings the inputs set (check: internal/loop TestLoopDriverCycle, internal/landing TestItemsGrowthIsChargedAcrossItsLandings, internal/plan TestSchedulerRankedByRealizedDelta)
- runs, regressions, failures and recurring costs -> durable obligations (check: internal/loop TestCycleFilesDeclinesAndAcceptsNothing, cmd/loop TestScorecardGapsBecomeProposals, cmd/validate TestALaneTestThatFailsIsRecordedBeforeTheNextPackageEnds)
- a measurement, or no action, = a valid outcome (check: internal/loop TestCycleFilesDeclinesAndAcceptsNothing, cmd/gate TestSmallGateLandsAChange)
- rank by recorded outcomes and expected cost (check: internal/plan TestSchedulerRankedByRealizedDelta)
- exploration = bounded (check: internal/agentloop TestCoordinatorBoundsSessionSteps, internal/loop TestLoopDriverCycle)
- rejection feeds the next decision (check: internal/loop TestLoopDriverCycle, internal/loop TestLoopAcceptedPrerequisiteProgress)
- missing resource => explicit resume condition and continue independent work (check: cmd/validate TestACellWhoseRuntimeIsNotHereWaitsOnItsLane, internal/plan TestDispatchPassesRowsAwaitingADecision, internal/landing TestALandingWaitingOnARunLeavesTheCheckoutFree)
- persist position, dispositions, outstanding checks and budget => resume without duplicates (check: cmd/loop TestInterruptedCycleResumesAtItsFirstUnpassedPhase, cmd/validate TestCellCreditSurvivesAKilledRun, cmd/loop TestScorecardGapsBecomeProposals)
- freeze acceptance before the work (check: internal/landing TestChangeIsJudgedByTheTermsItStartedWith, internal/scorecard TestADeclineKeepsItsKeyAndItsOriginalRestorationTarget)

## Working loop

1. campaign => `go run ./cmd/plan -next` (check: cmd/plan TestDispatchJSONMatchesProse, internal/plan TestEnforceCurrentFirstOpenStep)
   - honor dependencies, lanes, acceptance and stops (check: internal/plan TestDependencyDispatchSkipsBlocked, internal/plan TestLaneOwnedDispatch, cmd/plan TestStopStatusProjection)
   - preserve others' work (check: internal/landing TestTrunkCheckoutFollowsWithItsOwnChangesKept, internal/landing TestLandingLeavesAConcurrentPlanEditAlone, cmd/loophook TestStopIgnoresPreexistingDirt)
2. state the change, its expected effect and the deciding comparison (check: internal/plan TestOpenStepRequiresVerifier, cmd/gate TestSmallGateLandsAChange)
3. preflight first = `go run ./cmd/gate -plan <item>/<step> -message-file - -preflight` (check: internal/landing TestPreflightRunsTheCheapStagesOnly, internal/landing TestLandingRecordsItsWaitsByCause)
   - after a hand edit => `go run ./cmd/validate -claims` (check: cmd/validate TestPublishedMatricesAreCurrent, cmd/validate TestGenerateRejectsStaleClaimEvidence)
4. land = `go run ./cmd/loop -land <item>/<step> -message-file -` (check: cmd/loop TestRowSequenceStateMachine, cmd/loop TestLandReadsStructuredVerdict, cmd/plan TestPromptContractProjection)
   - message carries `Cause:` and `Predicted effect:` (check: cmd/gate TestSmallGateLandsAChange)
   - the gate writes the numbers (check: cmd/gate TestSmallGateLandsAChange, internal/landing TestLandingSaysWhatAChangeAddsOfEachKind, internal/landing TestLandingMeasuresTheScorecardSlice)
   - landed <=> `git log -1` shows `Landed-Row:` (check: cmd/gate TestSmallGateLandsAChange, cmd/loop TestRowSequenceStateMachine, internal/plan TestAcceptedProgressReadsLandingsFromGit)
   - never a raw commit (check: internal/guard TestGuardCorpus)
   - merge = `git merge --no-ff --no-commit <branch>` -> resolve -> land (check: internal/landing TestProgressSeesALandedMerge)
5. open decision and no other row ready => its row goes on under the recommended option, an answer given before then taken instead (check: internal/plan TestAQuestionNeverHoldsTheLoop, internal/plan TestAnswerTakesOneOfTheOptions)

- landing returns to dispatch (check: internal/loop TestLoopContinuationAfterCommit, cmd/loophook TestStopDecision)
- pause <=> an explicit stop request (`go run ./cmd/plan -stop <reason>:<detail>`, its reasons in the command's help) (check: cmd/plan TestEnforceStopReason, cmd/loophook TestStopDecision, cmd/loop TestPersistentLoopStop)
- lift a pause = `-resume-stop` (check: cmd/plan TestStopStatusProjection, cmd/loophook TestPersistentStopHook)
- prompt => answer -> continue (check: cmd/loophook TestBoundedRequestScopesAnswerNotTurn, cmd/loophook TestStopDecision)
- question => answer (check: cmd/loophook TestBoundedRequestScopesAnswerNotTurn)
- instruction => work (check: cmd/loophook TestBoundedRequestScopesAnswerNotTurn)
- one committing session per checkout (check: cmd/loophook TestSecondImplementingSessionIsSentToAWorktreeOfItsOwn, cmd/loophook TestOnlyAnImplementingSessionTakesTheHold)
- another session => its own worktree and the commit queue, or review only (check: cmd/loophook TestSecondImplementingSessionIsSentToAWorktreeOfItsOwn, internal/landing TestTwoWorktreesLandThroughOneQueue, internal/guard TestReviewSessionEditsNothingTracked)
- never touch another session's work (check: cmd/loophook TestSecondImplementingSessionIsSentToAWorktreeOfItsOwn, internal/guard TestReviewSessionRemovesNothing, cmd/loophook TestOverlappingSessionsKeepTheirOwnTurnSnapshots)
- `go run ./cmd/loop` supervises unattended work (`.overgo-runtime/loop.json`) (check: internal/loop TestLoopDriverCycle, cmd/loop TestLoopConfigResolvesStrategyAuthority)
- one lock per worktree (check: internal/processcontrol TestCampaignRunningFollowsOwnerLock, internal/authoritylock TestAcquireSerializesPlanAndGateMutation)
- off-scope finding => `go run ./cmd/finding` (check: cmd/finding TestRecordFindingRequiresVerifierAndUsesTypedStore, cmd/plan TestPromptContractProjection)

### Gate and cycle

- the gate judges the commit from the tree alone (unit, golden, hermetic, small device) (check: internal/landing TestReachInvariants, internal/landing TestLandingPublishesTheCheckedCandidate)
- a store write never changes the gate's verdict (check: internal/landing TestReachInvariants)
- the cycle (`validate -run`) judges models (real-model, live-store and device suites, by build tag) (check: internal/landing TestAudioLaneOptsIntoTheStoreAcceptance, internal/landing TestReachInvariants, cmd/validate TestValidateRunCycleLanes)
- cycle phases = preflight -> lanes -> cells -> publish (check: cmd/loop TestChecksRunBeforeTheCells, cmd/loop TestCycleReportsWhatItCreditedAndEachPhaseWall, cmd/loop TestARunRecordsItsEnd)
- a check a long run depends on runs before it, never after, since one failing after spends the run => a failed lane ends the cycle before its first cell (check: cmd/loop TestChecksRunBeforeTheCells)
- a cell its preflight refuses => named before any cell is measured, and the others are measured (one model's missing precondition leaves the others' numbers sound) (check: cmd/validate TestPreflightRefusesAnUnmeetableCell, cmd/loop TestChecksRunBeforeTheCells)
- preflight = each cell's own producer run to where its measurement begins (`clioptions.Ready`) => a precondition is checked by the code that needs it, never by a copy of its rule (check: cmd/validate TestPreflightRefusesAnUnmeetableCell, cmd/evaluate TestAQualityCellsPreflightReachesItsSetup)
- interrupted cycle => resumes at the first unpassed phase when `go run ./cmd/loop -improve` runs again (check: cmd/loop TestInterruptedCycleResumesAtItsFirstUnpassedPhase, cmd/loop TestUnfinishedCycleContinuesAtItsOwnCommit, cmd/validate TestCellCreditSurvivesAKilledRun)
- cycle running => host work goes on: its measurement and fill hold the host's lane turn, so a landing's lanes wait for the device in code (`landing.HoldLaneTurn`), and its record says when it ends (`.overgo-runtime/measurement_run.json`, printed by the hooks) (check: internal/landing TestALandingBesideARunWaitsOnlyForTheDevice, internal/landing TestALandingWaitingOnARunLeavesTheCheckoutFree, cmd/loop TestARunRecordsItsEnd)
- timing taken beside other work != a measurement (check: cmd/validate TestAContendedCellIsMeasuredOnceMoreAfterTheOthers, internal/landing TestTimedWorkExcludesCompetingWork, internal/landing TestATimedMeasurementAndALandingNeverRunTogether)
- gate's first stage = `space` => refuses below a free-space floor on the temporary directory's volume (check: internal/landing TestCommitChecksRefuseBelowTheFreeSpaceFloor)
- `space` applies to preflights and plan-only landings too (check: internal/landing TestCommitChecksRefuseBelowTheFreeSpaceFloor)
- `space` refusal = external => free space -> land again (check: internal/landing TestCommitChecksRefuseBelowTheFreeSpaceFloor)
- landing grows its row past its count => raises the count with a reason in its own plan edit (gate rule) (check: internal/landing TestLandingRaisesItsOwnCount, internal/landing TestItemsGrowthIsChargedAcrossItsLandings, cmd/plan TestSetCountWritesTheCountAndItsReason)
- filing a check => run it once (`plan -verify`): it fails as a test or names a test not yet written, never vacuous by its form (a pipe, a wrong package), because a landing is judged by the check its row held before it (check: cmd/plan TestRunVerifyRejectsSkippedGoTest, internal/testevidence TestVerifyGoTestEvidencePairsSelectorsPerInvocation, internal/landing TestChangeIsJudgedByTheTermsItStartedWith)
- land before an evaluation run (check: internal/runrecord TestVerifyingCommitRequiresCleanWorktree, cmd/loop TestCycleRunsInItsOwnCheckout)
- native `go test` selects (check: cmd/gate TestSelectionAnalyzersGone)
- lane's selection = every package whose test binary compiles a changed package, read from what the binary compiles (all its test files, every package under a recursive pattern), never from a proxy (production imports, a pattern's top directory) (check: internal/landing TestLaneReachSkipsOtherPackagesTests, internal/landing TestLaneSelectionUsesTheLaneBuildContext, internal/landing TestReachInvariants)
- never time budgets, ratchets or a second selector (check: internal/repoanalysis TestNoFixedTimers, cmd/gate TestOldGateMachineryGone, cmd/gate TestSelectionAnalyzersGone)

### Plan and SQA

- `docs/plan.json` = sole dispatch authority (check: internal/plan TestSinglePlanAuthority)

## Evidence

- bind source, model, data, recipe, policy and environment identities exactly (check: internal/evaluation TestInstructionResumeRejectsDifferentBindings, internal/evaluation TestEvaluationEvidenceBindsPlanShardsAndAuthorities, internal/runrecord TestVerifyGateRunRejectsUnboundIdentities)
- reuse or resume <=> inputs and contracts match (check: internal/evaluation TestAnswersOutliveOnlyTheCommitsThatLeaveTheirCodeUnchanged, internal/capabilityruntime TestStoredResultIsReusedOnlyUnderItsOwnIdentity, cmd/validate TestCellCreditSurvivesAKilledRun)
- one run serves every obligation with matching identity and contract (one record, many readers) (check: internal/evaluation TestEvaluationAtAnUnchangedPlanReusesItsCampaign, cmd/loop TestMeasurementRunsNoCaseAgainAfterAnUnrelatedCommit, cmd/validate TestAcceptanceCoversARecipeReplacedForItsSession)
- a run never judges its own output (check: internal/runrecord TestTrainingDecisionWithholdsPromotionUntilEvaluation, internal/evaluation TestCommittedVerdictRescoresItsReports)
- a run's evidence = complete by its own contract (check: internal/evaluation TestResumeRejectsPartialOrForeignPlanResults, internal/testevidence TestVerifyGoTestEvidenceWithoutRun, internal/testevidence TestCompleteRunRecordsDeclaredExclusions)
- unknown provenance => keeps its value and earns no compatible evidence (check: internal/evaluation TestEvaluationExecutionDoesNotInventMissingAuthorities, cmd/validate TestBenchmarkDoesNotReuseUnknownExecution)
- replay carries only bit-identical outputs (check: internal/capabilityruntime TestReplayDoesNotCrossACodeChange, internal/evaluation TestAnswersOutliveOnlyTheCommitsThatLeaveTheirCodeUnchanged)
- evaluate decides (check: internal/runrecord TestTrainingDecisionWithholdsPromotionUntilEvaluation, internal/evaluation TestHeldoutVerdictRequiresStrictGainOnDeclaredMetric)
- compare identical tasks, artifacts, protocols and budgets (check: internal/scorecard TestTwoSuitesAreNeverOneMove, internal/scorecard TestComparisonIsTyped, internal/scorecard TestALongerPromptIsTimedAsItsLengthAlone)
- difference inside the rounds' ranges = no difference (check: internal/scorecard TestADeclineMustPassWhatItsReferenceAllows, internal/scorecard TestACellsOwnSpreadNeitherHidesItsDeclineNorRestoresIt)
- an A/B => its failed requests counted beside its rates, because a pool of the requests that succeeded hid a build that corrupted the card's experts (14 requests failed, its rates within the others' ranges) (check: internal/scorecard TestServingCountsWhatUsersHit, internal/scorecard TestADeclineMustPassWhatItsReferenceAllows)
- aggregate gain never hides a required regression (check: internal/evaluation TestMoERoutingPromotionRejectsHiddenStratumOrCostRegressions, internal/scorecard TestComparisonIsTyped)

## Robustness

- persist results and obligations (check: internal/runrecord TestCancelledGatePersistsTerminalTruth, cmd/validate TestCellCreditSurvivesAKilledRun, internal/loop TestObligationCompletionReplay)
- publish and retry = idempotent (check: internal/overgodb TestCommitIsIdempotentByKeyAndContent, internal/agenttool TestPublishManualCatalogIsAtomicAndIdempotent, internal/repoanalysis TestIdempotentCommitHasOneOwner)
- recover from durable state (check: internal/workflowruntime TestWorkflowRestartSkipsCompletedStages, internal/overgodb TestCheckpointGenerationsSurviveATornWrite, cmd/loop TestInterruptedCycleResumesAtItsFirstUnpassedPhase)
- retried failed write never becomes a pass (check: internal/evaluation TestInstructionResumeRetainsQualityFailure, internal/evaluation TestInstructionResumePublicationFailure, internal/overgodb TestAppendFaultClosesEveryStoreSurfaceUntilReopen)
- one home a resource (check: internal/repoanalysis TestEntryAuthoritiesHoldOverTheTree, internal/repoanalysis TestModelOpensShareTheClaimedLoad, internal/repoanalysis TestReleaseIsTheOnlyRetentionWriter)
- bound workers (check: internal/capabilityruntime TestBoundedSessionAdmission, internal/capabilityruntime TestSessionDirectorBoundsComponentWaitingAndReportsLifecycle)
- handle cancellation, error, death and release (check: internal/processcontrol TestDeterministicCancellationContracts, internal/processlock TestAcquireReleasedAfterProcessDeath, internal/processlock TestAcquireCancellationReleaseRace)
- OS locks only (check: internal/processlock TestAcquireRejectsFalseContention, internal/processlock TestAcquireReleasedAfterProcessDeath, internal/processcontrol TestCampaignRunningFollowsOwnerLock)
- never steal a lock (check: internal/processlock TestAcquireIsExclusiveAndReleasedByClose, internal/processlock TestAcquireContextContentionIdentity, internal/processcontrol TestHoldResourceAdmitsOneHolderAlone)
- model, dataset and checkpoint inputs = immutable (check: internal/pytorchzip TestRecordNamesWhatAnArchiveRecords, internal/latentvideo TestPromptRowsFollowTheBytesTheyAreMadeFrom, internal/modelartifact TestRawTextPublicationPreservesImmutableFacts)
- publish through the lifecycle's home (check: internal/repoanalysis TestEntryAuthoritiesHoldOverTheTree, internal/repoanalysis TestIdempotentCommitHasOneOwner, internal/repoanalysis TestReleaseIsTheOnlyRetentionWriter)
- before retiring originals => validate rollback (check: cmd/overgodb TestFailedCheckNeverSwaps, cmd/overgodb TestFailedReplacementKeepsTheLivePath, internal/overgodb TestRSIStoreMigrationDrill)
- execution, verification and activation = separate (check: internal/modelrecipe TestActivateCapabilityRequiresBoundVerification, internal/agenttool TestCandidateCatalogStagesVerifiesAndActivatesExactly, internal/modelrecipe TestAtomicEvidenceGatedActivation)
- keep rollback predecessors (check: internal/overgodb TestPromotionReversibleAndDrillRestoresChampion, cmd/overgodb TestFailedReplacementKeepsTheLivePath)
- expose missing, invalid and partial state (check: internal/scorecard TestACellNotMeasuredThisRunSaysSo, internal/discovery TestServableReportsStaleActivationsWithoutFailing, internal/evaluation TestResumeRejectsPartialOrForeignPlanResults)
- honor guards and authorization (check: internal/guard TestGuardCorpus, internal/server TestCrossOriginMutationProtection, internal/agentloop TestCoordinatorGatesMutationBehindInspectionAndApproval)
- never a silent reset, a semantic change or a weakened acceptance (check: internal/landing TestChangeIsJudgedByTheTermsItStartedWith, internal/trainingworkflow TestSelectionNeverRestoresEligibility, internal/scorecard TestADeclineKeepsItsKeyAndItsOriginalRestorationTarget)
- failing check => fixed, never relabeled (check: internal/landing TestChangeIsJudgedByTheTermsItStartedWith, cmd/plan TestRunVerifyRejectsSkippedGoTest, internal/repoanalysis TestIntegrationSkipReasonNamesItsGate)
- no clock decides failure => signals decide (check: internal/repoanalysis TestNoFixedTimers, internal/processcontrol TestResourceAdmissionWakesOnHolderExit, internal/processcontrol TestDeterministicCancellationContracts)

## Efficiency

- measure whole attempts (proposal, failure, wait, acquisition, verification, recovery) (check: internal/landing TestLandingRecordsItsWaitsByCause, internal/runrecord TestServingAttemptChainClassifiesRetryReselectionAndSpillover, cmd/loop TestCycleReportsWhatItCreditedAndEachPhaseWall)
- elapsed time != summed parallel time => report them apart (check: internal/processmeasure TestStagesRefuseOverlappingParts, internal/processmeasure TestStagesDivideARequest, internal/landing TestLanesMeasurementLine)
- count repeated loads (check: internal/capabilityruntime TestSessionDirectorBoundsComponentWaitingAndReportsLifecycle, internal/capabilityruntime TestMeasurementCarriesTheLoadWall)
- reuse valid artifacts, checkpoints and check results (check: internal/capabilityruntime TestStoredResultIsReusedOnlyUnderItsOwnIdentity, cmd/validate TestCellCreditSurvivesAKilledRun, internal/evaluation TestEvaluationAtAnUnchangedPlanReusesItsCampaign)
- reacquire <=> invalidated (check: cmd/validate TestResumedRunStartsAgainACellWhoseWeightsChanged, internal/discovery TestMemoReusesOnlyUnchangedFiles, internal/capabilityruntime TestStoredResultIsReusedOnlyUnderItsOwnIdentity)
- selection, ordering, reuse and reporting = from typed facts (AST, types, parsed commands, declared capabilities) (check: internal/repoanalysis TestNoRegexParsesStructuredData, internal/landing TestLaneSelectionUsesTheLaneBuildContext, internal/landing TestLaneReachSkipsOtherPackagesTests)
- external text => normalize at the boundary (check: internal/executionfailure TestFailureNormalizationContract, internal/evaluation TestInstructionGenerationWithAnInvalidByteResumes, cmd/kernel-manifest TestFileHashCanonicalizesCheckoutNewlines)
- scope verification through the dependencies' homes (check: internal/devicescope TestDeviceFunctionImpact, internal/devicescope TestPlanForReachesImportingTests, internal/landing TestLaneReachSkipsOtherPackagesTests)
- unknown dependency => keep the broader scope (check: internal/devicescope TestDeviceConservativeFallback, internal/landing TestLaneReachSkipsOtherPackagesTests)
- conflicting mutations and isolated measurements => serialized (check: internal/landing TestATimedMeasurementAndALandingNeverRunTogether, internal/authoritylock TestAcquireSerializesPlanAndGateMutation, internal/landing TestCommitsOfTwoWorktreesTakeTurns)
- CPU work holds no GPU claim (check: internal/landing TestCPULanesOverlapTheDeviceLane, internal/landing TestALandingWaitingOnARunLeavesTheCheckoutFree)
- remove unused flags, wrappers, state and exports (check: internal/landing TestExportUseCountsConsumers, internal/repoanalysis TestEveryProcessDeclarationIsReachedByACommand, internal/repoanalysis TestProjectorSessionOwnsNoPromptWrappers)
- never rerun for a favorable result (check: internal/evaluation TestInstructionResumeRetainsQualityFailure, internal/scorecard TestOneBusyAttemptCannotMaskADecline)

### Speed work

- A/B = one loaded model, forced token ids, interleaved arms > a reload and free-running text an arm (check: cmd/benchmark TestPairedComparison)
- a buffer sized to a run's need => rounded up before it grows, because each growth frees it, drops what names it and waits while Windows clears the new memory (about 100 ms at 2.8 GB), and a share to spare costs what that memory would hold (experts) (check: internal/cuda/executor TestArenaMetrics, internal/cuda/executor TestDeviceBufferClass)
- a tuning value (a threshold, a size or a count that trades measured costs: a chunk's tokens, a tile's rows, an expert's admission) => its comment names the cost it minimizes and the measured inputs it is computed from (the link's rate, the bytes a token holds, an expert's bytes, a miss's cost, the request's tokens), because a sweep returns only a point it tried, for one request shape on one machine (2,048-token chunks paid for their slower read after about 700 decoded tokens) (check: internal/repoanalysis TestTuningValuesNameTheirCost)
- memory a reference frees for another use => taken where the reference takes it (its coldest slots, across layers), because freeing by allocation order took whole layers' hot experts (a release now keeps a freed allocation's hotter experts in colder ones' slots) (check: internal/modeldevice TestReleaseFreesTheColdestClass, internal/modeldevice TestReleaseKeepsTheHottestExperts)
- memory moved between two uses (a read's arena and the experts) => lent, the part the next use needs taken back, never freed and allocated again, because each allocation waits while Windows clears it and each free moves experts out (freed, a 16K read's arena and the expert releases added about 220 ms to its first token) (check: internal/cuda/executor TestPromptMemoryGivesBackItsLoan, internal/cuda/executor TestDeviceBufferPoolLendsItsFreeBuffers, internal/inference TestAcceleratorTeardownReclaimsPromptLoans)
- measured gains of one row => one landing (steps joined by commas) > a gate run each (check: internal/landing TestLandingCompletesSeveralSteps)

## Compute and derived values

- installed capacity != budget (check: internal/server TestResidentFitAdmitsADeviceRunOnlyBesideTheServedModel, internal/modeldevice TestExpertResidencyFillsTheCardWhenItFrees)
- placement, batch, chunk, workers and residency = derived from workload, measurement and availability (check: internal/cuda/executor TestHostSlotLimitTradesACopyForTheCPU, internal/modeldevice TestExpertResidencyFillsTheCardWhenItFrees, internal/hostmath TestDispatchSchedulingPreservesResults)
- scale down by streaming, tiling and bounded concurrency (check: internal/composition TestOfflinePlanDerivesResourcePolicy)
- scale up <=> measured benefit (check: internal/hostmath TestDispatchSchedulingPreservesResults)
- preserve numerical contracts and checkpoint identity (check: internal/model TestBlockSelectionDoesNotDependOnChunk, internal/artifact TestHasherMatchesIdentifyForAnyChunking, internal/trainingprogram TestResumeSidecarRoundTripsBitExact)
- nothing fits => resumable wait or reported limit (check: internal/processcontrol TestResourceAdmissionWaitsForRelease, internal/server TestResidentFitAdmitsADeviceRunOnlyBesideTheServedModel)
- number other than 0 or 1 => named constant with a reason (gate rule) (check: internal/landing TestNewPolicyLiteralsIgnoresRenames)
- thresholds, tolerances, sample sizes and optimizer settings = derived from declarations or observations or math (check: internal/trainingprogram TestBuiltinOptimizerPolicyDerivesConfig, internal/server TestDefaultNeighborCountIsDerived)
- limit with no derivation = arbitrary literal => replace it with a rule (check: internal/landing TestNewPolicyLiteralsIgnoresRenames, internal/repoanalysis TestNoFixedTimers, internal/composition TestOfflinePlanRejectsLiteralPolicy)
- kept value => record its source, scope and reconsideration condition in the owning contract (check: internal/composition TestOfflinePlanRejectsLiteralPolicy, internal/repoanalysis TestTuningValuesNameTheirCost)
- distribution, geometry or metric assumption (Gaussian, independent, Euclidean, linear interpolation) => justified per model and task (check: internal/evaluation TestNumericTargetEvaluationAvoidsShapeAndDistributionAssumptions, internal/server TestAnalyzeStatesDefaultsToSpearman, internal/server TestAnalyzeTensorsReturnsDistributionFreeProfiles)

## Training

- rates = derived (check: internal/latentvideo TestDiTTrainerDerivedHyperparameters, internal/trainingprogram TestBuiltinOptimizerPolicyDerivesConfig, internal/routedlm TestFlowHeadTrainerDescendsAndDerivesHyperparameters)
- policy = `internal/trainingprogram/optimizer_policy.json` (check: internal/trainingworkflow TestBootstrapTokenRecipeBindsStoredOptimizerPolicy, internal/trainingprogram TestOptimizerPolicyRequiresRecipeContent)
- rate = P^exponent (check: internal/trainingprogram TestBuiltinOptimizerPolicyDerivesConfig, internal/trainingprogram TestOptimizerPolicyRecordedBeforeInitializationServesBoth)
- exponent = -0.6 for a trained model (check: internal/trainingprogram TestDerivedRateFineTunesMeasuredModels, internal/adaptiveparity TestQwen35RealTraining)
- exponent = -0.5 from initialization (check: internal/scratchmodel TestScratchConstructionAuthority, internal/trainingprogram TestOptimizerPolicyRecordedBeforeInitializationServesBoth)
- momentum from N=30 (check: internal/latentvideo TestDiTTrainerDerivedHyperparameters, internal/routedlm TestFlowHeadTrainerDescendsAndDerivesHyperparameters)
- optimizer math on device (check: internal/hybridtrain TestHybridTrainDeviceResidentMatchesHost, internal/optimizer TestDeviceMuonStepPlanMatchesHost)
- only scalars cross to host (check: internal/hybridtrain TestHybridTrainDeviceResidentMatchesHost, internal/adaptiveparity TestQwen35RealTraining)
- `cmd/train` = the one entry (`-route`, `-objective`) (check: cmd/train TestTrainRoutesEveryProbe, cmd/train TestTrainRoutesFoldedProbes)
- quantized format => native fast path > output identity with bf16, guarded by held-out quality (check: internal/evaluation TestKQuantProductsHoldHeldOut, internal/evaluation TestQwen35CellsHoldHeldOut)

## Implementation

- Go owns runtime, orchestration and tooling (check: internal/repoanalysis TestRSIRuntimeIsGoOnly, internal/repoanalysis TestEntryAuthoritiesHoldOverTheTree)
- never Python in the tree (check: internal/repoanalysis TestPythonLivesOnlyInItsIgnoredPlace)
- goldens = data from the Python original (check: internal/repoanalysis TestPythonLivesOnlyInItsIgnoredPlace, internal/latentimage TestMOSSGoldenProvenance)
- CUDA through the kernel manifest (check: cmd/kernel-manifest TestKernelManifestIsCurrent, cmd/kernel-bindings TestGeneratedBindingsMatchManifest, internal/cuda/kernel TestValidateAssets)
- scratch -> `tmp/<row id>/`, the folder named for the plan row that made it (check: cmd/loop TestTmpEntryBelongsToAnOpenRowOrIsFinished)
- executables -> `bin/` (check: cmd/release TestReleaseVersionAndArchiveLayout, cmd/release TestBinHygiene)
- Python lives only in `scripts/python_golden_references/` (git-ignored) (check: internal/repoanalysis TestPythonLivesOnlyInItsIgnoredPlace)
- the gate and the Go code never run those scripts (check: internal/repoanalysis TestEntryAuthoritiesHoldOverTheTree, internal/repoanalysis TestRSIRuntimeIsGoOnly)
- models, datasets, store and `build/` live in `OVERGO_DATA_ROOT`, outside the module (check: internal/dataroot TestEnvBaseWinsAndNamesAllRoots)
- llama.cpp for pinned behavior (check: internal/quant TestQuantizeMatchesPinnedGGMLReference, internal/hostexpert TestQuantizersRoundAsGGML, internal/tokenizer TestUpstreamTokenizerCorpus)
- `adaptive_new` = read-only (check: internal/repoanalysis TestNoTestReadsAdaptiveNew)
- keep commits, licenses and artifact identities (check: internal/runrecord TestExternalMechanismProvenanceIsExactAndNoRuntimeDependencyIsAdmitted, internal/latentimage TestMOSSGoldenProvenance)
- model-family facts live in artifacts and recipes, never in model-specific code (check: internal/inferencesurface TestServingCodeNamesNoArchitecture, internal/model TestGemma4ProgramUsesNeutralStages)
- clear errors and cancellation (check: internal/processcontrol TestDeterministicCancellationContracts, internal/workflowruntime TestCancellationCauseReachesTerminalReceipt)
- reusable hot-path buffers (check: internal/audiodsp TestFrameTraceSteadyStateAllocations, internal/media TestResamplePolyReusesScratchAndRejectsOverlap)
- `gofmt` and `go vet` (check: internal/landing TestPreflightRunsTheCheapStagesOnly, internal/repoanalysis TestGoStyleCensus)
- numerical or gradient change => reference comparison (check: internal/latentvideo TestDiTTrainerGradientMatchesFiniteDifference, internal/routedlm TestModalityTransformerGradientMatchesFiniteDifference, internal/optimizer TestDeviceMuonStepPlanMatchesHost)

### Cleanup

- cleanup = deterministic code (check: cmd/loop TestLeftoversListsEveryDeclaredPlaceWithItsRule, cmd/loop TestCommitClearsWhatNothingHolds)
- cleanup != a model's judgement and != a question to a person (check: cmd/loop TestCommitClearsWhatNothingHolds, cmd/loop TestLeftoversDeletesOnlyWhatItsTableMarksFinished)
- every place => a rule that ends in keep or delete (check: cmd/loop TestLeftoversListsEveryDeclaredPlaceWithItsRule)
- entry kept <=> named by the store or a committed file or an open plan row or a running process (check: cmd/loop TestTmpEntryBelongsToAnOpenRowOrIsFinished, cmd/loop TestLeftoversListsEveryDeclaredPlaceWithItsRule, cmd/loop TestLeftoversDeletesOnlyWhatItsTableMarksFinished)
- entry not kept and its run's record landed => deleted (check: cmd/loop TestCommitClearsWhatNothingHolds, cmd/loop TestHelperCopyIsRemovedOnceItsCommitIsOnTheTrunk)
- command that makes leftovers => deletes them when it writes its record (check: cmd/loop TestCommitClearsWhatNothingHolds, cmd/validate TestTrainingAcquisitionRetainsPrivateScratchUntilCompletion, cmd/loop TestLandReadsStructuredVerdict)
- decision outside the system's authority (a branch is retired) => recorded once as data -> code acts on it (check: internal/plan TestAnswerTakesOneOfTheOptions, internal/plan TestDispatchPassesRowsAwaitingADecision)
- cleanup request met <=> bytes or files or code are gone (check: cmd/loop TestHelperCopyIsRemovedOnceItsCommitIsOnTheTrunk, cmd/loop TestLeftoversDeletesOnlyWhatItsTableMarksFinished)
- campaign close-out refused <=> a deletable entry remains (check: cmd/loop TestCloseoutRefusesFinishedLeftovers)
- fixture => named by a test (check: internal/repoanalysis TestEveryFixtureHasAReader)

## Verification

- missing, skipped, incomplete or zero-match check => earns no pass (check: internal/testevidence TestVerifyGoTestEvidenceWithoutRun, cmd/plan TestRunVerifyRejectsSkippedGoTest, internal/plan TestOpenStepRequiresVerifier)
- keep negative and inconclusive results (check: internal/runrecord TestCancelledGatePersistsTerminalTruth, cmd/validate TestALaneTestThatFailsIsRecordedBeforeTheNextPackageEnds, cmd/loop TestLeftoversListsEveryDeclaredPlaceWithItsRule)
- report denominators, exclusions, scope and environment (check: internal/scorecard TestHeldoutReadingStatesItsCasesAndNoise, internal/testevidence TestCompleteRunRecordsDeclaredExclusions, internal/scorecard TestServingEnvironmentChangeIsUnknown)
- real models one at a time, smallest first (check: cmd/validate TestSelectValidation, internal/discovery TestServableReportsStaleActivationsWithoutFailing, internal/landing TestATimedMeasurementAndALandingNeverRunTogether)
- tools = `cmd/validate` (`-scorecard`, `-run`, `-claims`, `-tests`, `-smoke`, `-parity`, `-receipt`), `cmd/device-lane`, `cmd/webui-lane` (check: internal/repoanalysis TestRootDocumentsNameOnlyWhatTheTreeHolds, cmd/validate TestParitySubcommandsReplaceTheirCommands)

## Sources of truth

| Surface | Holds |
| --- | --- |
| [skill.md](skill.md) | The system's doctrine. Sole root agent-instruction document |
| [docs/plan.json](docs/plan.json) | Rows, dependencies, acceptance, open work, and rules that bind one campaign only |
| OvergoDB | Artifacts, lineage, runs, decisions, findings, receipts (`go run ./cmd/overgodb query`) |
| [compatibility.json](compatibility.json) | Model and capability verification contracts |
| [kernels/manifest.json](kernels/manifest.json) | Kernel ABI, source and binary identities |
| [API manifest](docs/api_manifest.json) | Public commands, routes, document contracts |

- existing homes only (check: internal/repoanalysis TestEntryAuthoritiesHoldOverTheTree, internal/repoanalysis TestAliasMovesHaveOneOwner, internal/repoanalysis TestIdempotentCommitHasOneOwner)
- never parallel plans, ledgers, schedulers or recovery paths (check: internal/plan TestSinglePlanAuthority, cmd/gate TestClosureLedgerGone, internal/repoanalysis TestReleaseIsTheOnlyRetentionWriter)
- contracts in docs, measurements in the store, history in Git (check: internal/plan TestAdvanceRemovesCompletedRows, internal/plan TestAcceptedProgressReadsLandingsFromGit, cmd/validate TestScorecardHasNoUnexplainedCell)
- report = outcome + material measurements + checks + remaining obligations (check: cmd/loop TestARunRecordsItsEnd, cmd/loop TestCycleReportsWhatItCreditedAndEachPhaseWall)
