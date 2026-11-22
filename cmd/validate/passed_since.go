package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/landing"
	"overgo/internal/plan"
)

// gitSince is whether, in the repository at root, the commit later is earlier or one descending from it (git merge-base
// --is-ancestor), each pair asked once; a pair git cannot read reads as no.
func gitSince(ctx context.Context, root string) func(earlier, later string) bool {
	answers := map[[2]string]bool{}
	return func(earlier, later string) bool {
		pair := [2]string{earlier, later}
		if answer, read := answers[pair]; read {
			return answer
		}
		_, err := gitauthority.Query(ctx, root, "merge-base", "--is-ancestor", earlier, later)
		answers[pair] = earlier == later || err == nil
		return answers[pair]
	}
}

// answerPassedSince answers -passed-since: the commit named, resolved in the repository at root, against the passes the
// trunk keeps (testsPassedSince).
func answerPassedSince(ctx context.Context, output io.Writer, root, named string, tests []string) error {
	commit, err := gitauthority.Query(ctx, root, "rev-parse", "--verify", named+"^{commit}")
	if err != nil {
		return fmt.Errorf("-passed-since %s names no commit: %w", named, err)
	}
	passes := plan.ReadTestPasses(cmp.Or(trunkCheckout(ctx, root), root))
	return testsPassedSince(output, passes, strings.TrimSpace(string(commit)), tests, gitSince(ctx, root))
}

// answerLanesPassedAt answers -lanes-passed-at: the commit named, resolved in the repository at root, against the lanes'
// records the trunk keeps (lanesPassedAt).
func answerLanesPassedAt(ctx context.Context, output io.Writer, root, named string) error {
	commit, err := gitauthority.Query(ctx, root, "rev-parse", "--verify", named+"^{commit}")
	if err != nil {
		return fmt.Errorf("-lanes-passed-at %s names no commit: %w", named, err)
	}
	runs := plan.ReadLaneRuns(cmp.Or(trunkCheckout(ctx, root), root))
	return lanesPassedAt(output, runs, landing.CycleLanes(), strings.TrimSpace(string(commit)))
}

// lanesPassedAt answers, from the lanes' records (plan.RecordLaneRun, plan.RecordLanePackages), whether one whole slow
// run passed at commit (slow-whole): every cycle lane's last pass, and each package's of a lane run package by package,
// is of that commit. A run cut to what changed passes a lane at its commit while the packages it cut keep passes of
// earlier ones, and a lane it reached nowhere keeps its own, so each is read. It says each lane's answer on output, and
// is an error naming every lane and package whose last pass is of another commit or none.
func lanesPassedAt(output io.Writer, runs map[string]plan.LaneRecord, lanes []landing.CycleLane, commit string) error {
	var other []string
	read := func(name string, pass plan.LaneRun) {
		switch {
		case pass.Commit == "":
			other = append(other, name)
			fmt.Fprintf(output, "%s: no pass kept\n", name)
		case pass.Commit != commit:
			other = append(other, name)
			fmt.Fprintf(output, "%s: last passed at %.12s on %s\n", name, pass.Commit, pass.Ended.Format("2006-01-02 15:04"))
		}
	}
	for _, lane := range lanes {
		record := runs[lane.Name]
		read(lane.Name, record.Passed)
		for _, arguments := range lane.Each {
			pkg := arguments[len(arguments)-1]
			read(lane.Name+" "+pkg, record.Packages[pkg])
		}
		if !slices.ContainsFunc(other, func(name string) bool { return name == lane.Name || strings.HasPrefix(name, lane.Name+" ") }) {
			fmt.Fprintf(output, "%s: passed whole at %.12s, %d packages\n", lane.Name, commit, len(lane.Each))
		}
	}
	if len(other) > 0 {
		return fmt.Errorf("-lanes-passed-at %.12s: %d lanes and packages last passed at another commit or none: %s", commit, len(other), strings.Join(other, ", "))
	}
	return nil
}

// testsPassedSince answers, from the passes the lanes and the landings keep (plan.RecordTestPasses), whether each test
// named <package>.<Test> last passed at commit or at one descending from it (since): it says each test's last pass on
// output, and is an error naming the tests with none since, so a step whose evidence is the slow tests' asks the tool
// that keeps their records (passes-query). flash-bound-read/read's check had been an awk over the lanes' log, which the
// gate's candidate checkout does not hold and whose lines it read wrongly three times (SQA-453).
func testsPassedSince(output io.Writer, passes map[string]plan.TestPass, commit string, tests []string, since func(earlier, later string) bool) error {
	if commit == "" || len(tests) == 0 {
		return errors.New("usage: validate -passed-since <commit> <package>.<Test>...")
	}
	var missing []string
	for _, named := range tests {
		at := strings.LastIndex(named, ".")
		if at <= 0 || !strings.HasPrefix(named[at+1:], "Test") {
			return fmt.Errorf("-passed-since: %q names no test as <package>.<Test>", named)
		}
		pass, held := passes[plan.PassedTest(named[:at], named[at+1:])]
		switch {
		case !held:
			fmt.Fprintf(output, "%s: no pass kept\n", named)
			missing = append(missing, named)
		case !since(commit, pass.Commit):
			fmt.Fprintf(output, "%s: last passed at %.12s on %s, before %.12s\n", named, pass.Commit, pass.Ended.Format("2006-01-02 15:04"), commit)
			missing = append(missing, named)
		default:
			fmt.Fprintf(output, "%s: passed at %.12s on %s\n", named, pass.Commit, pass.Ended.Format("2006-01-02 15:04"))
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("-passed-since %.12s: no pass kept at it or after it for %s", commit, strings.Join(missing, ", "))
	}
	return nil
}
