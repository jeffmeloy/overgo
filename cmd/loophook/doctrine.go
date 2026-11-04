package main

// doctrineText is the overgo turn contract emitted at SessionStart: the part of the doctrine a session needs before its
// first step, each rule citing the checks that hold it as skill.md's rules do (TestEveryDoctrineSentenceCitesItsCheck).
// skill.md is the doctrine, and its Judgement section holds what no code can.
const doctrineText = `OVERGO TURN CONTRACT. skill.md is the doctrine, its Judgement section holds what no code can, and each rule here cites the checks that hold it.

THE LOOP.
 1. ` + "`go run ./cmd/plan -next`" + ` names the step to work, and ` + "`go run ./cmd/plan -prompt`" + ` its contract: its check, its commit and its stops (check: cmd/plan TestDispatchJSONMatchesProse, cmd/plan TestPromptContractProjection).
 2. A step is done when its check passes: ` + "`go run ./cmd/plan -verify`" + ` runs it, and a check that skips a test fails (check: cmd/plan TestRunVerifyRejectsSkippedGoTest).
 3. A step lands only through ` + "`go run ./cmd/loop -land <item>/<step> -message-file -`" + `, the one way the trunk moves (check: internal/landing TestOnlyALandingMovesTheTrunkInGit, internal/landing TestALandingRefusesATrunkWithoutItsHook).
 4. The landing checks the dirty tree through the stages internal/landing's package doc lists, commits it with the step removed from the plan and ends in one verdict (check: internal/landing TestEveryStageFailureLeavesTheTrunkAlone, internal/landing TestLandingCompletesSeveralSteps, cmd/loop TestLandReadsStructuredVerdict).
 5. Its message states "Cause:" and "Predicted effect:", and one without either is refused (check: internal/landing TestAMessageWithoutItsSectionsIsRefused).
 6. A merge in progress (git merge --no-ff --no-commit <branch>, resolved) lands as the merge, its head joined as a parent (check: internal/landing TestProgressSeesALandedMerge).

STOPS.
 7. The Stop gate blocks a turn-end that leaves the turn's work uncommitted, and any turn-end while the plan holds an open row, except through its valves (check: cmd/loophook TestStopDecision, cmd/loophook TestAStopAfterWorkSinceItsBlockIsBlockedAgain).
 8. The one sanctioned pause is a recorded stop, ` + "`go run ./cmd/plan -stop <requested|irreversible|external-prereq>:<detail>`" + ` (check: cmd/plan TestEnforceStopReason, cmd/loophook TestStopDecision).

SESSIONS.
 9. A session whose first message opens with "review session" is held to no row (check: cmd/loophook TestReviewSessionIsNotHeldAndAnyOtherRoleIs).
 10. An implementing session holds its worktree from its first landing here until a turn of its ends with the plan stopped or complete and nothing uncommitted (check: cmd/loophook TestOnlyAnImplementingSessionTakesTheHold).
 11. A second implementing session in that worktree is sent to a worktree of its own, unless a message of its opens with "take over this worktree" (check: cmd/loophook TestSecondImplementingSessionIsSentToAWorktreeOfItsOwn).
 12. A session in a worktree of its own lands with the same command: its commit waits its turn among the worktrees, is brought up to the trunk, is checked there by the terms it started with and lands on the trunk, whose checkout follows (check: internal/landing TestCommitsOfTwoWorktreesTakeTurns, internal/landing TestWorktreeCandidateIsIntegratedOntoTheTrunk, internal/landing TestChangeIsJudgedByTheTermsItStartedWith, internal/landing TestTrunkCheckoutFollowsWithItsOwnChangesKept).
 13. adaptive_new is a reference tree to compare with or port from: no session writes into it, and no Go source names a path into it (check: internal/guard TestNoSessionWritesIntoTheReferenceTree, internal/repoanalysis TestNoTestReadsAdaptiveNew).`
