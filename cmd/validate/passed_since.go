package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
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

// answerPassedSince answers -passed-since: the commit named, resolved in the repository at root, against the results the
// trunk keeps (testsPassedSince).
func answerPassedSince(ctx context.Context, output io.Writer, root, named string, tests []string) error {
	commit, err := gitauthority.Query(ctx, root, "rev-parse", "--verify", named+"^{commit}")
	if err != nil {
		return fmt.Errorf("-passed-since %s names no commit: %w", named, err)
	}
	results := plan.ReadTestResults(cmp.Or(trunkCheckout(ctx, root), root))
	return testsPassedSince(output, results, strings.TrimSpace(string(commit)), tests, gitSince(ctx, root))
}

// answerLanesPassedAt answers -lanes-passed-at: the commit named, resolved in the repository at root, against the results
// the trunk keeps (lanesPassedAt), each one's changes read from its commit to the one named.
func answerLanesPassedAt(ctx context.Context, output io.Writer, root, named string) error {
	commit, err := gitauthority.Query(ctx, root, "rev-parse", "--verify", named+"^{commit}")
	if err != nil {
		return fmt.Errorf("-lanes-passed-at %s names no commit: %w", named, err)
	}
	at, since := strings.TrimSpace(string(commit)), gitSince(ctx, root)
	results := plan.ReadTestResults(cmp.Or(trunkCheckout(ctx, root), root))
	current := stillCurrent(root, results, func(base string) ([]string, error) {
		if !since(base, at) {
			return nil, fmt.Errorf("%.12s is not %.12s or before it", base, at)
		}
		listed, err := gitauthority.Query(ctx, root, "diff", "--name-only", base, at)
		return strings.Fields(string(listed)), err
	})
	return lanesPassedAt(output, results, landing.CycleLanes(), at, current)
}

// lanesPassedAt answers, from the results the lanes keep (plan.RecordTestResults), whether the slow tests passed at commit
// (slow-whole): each lane run whole has a pass current at it, and each test of a lane run test by test whose result is
// kept is a pass current at it (current, stillCurrent read up to commit). It says each lane's answer on output, and is
// an error naming every lane that does not.
func lanesPassedAt(output io.Writer, results map[string]plan.TestResult, lanes []landing.CycleLane, commit string, current func(lane, pkg, key string) bool) error {
	var other []string
	for _, lane := range lanes {
		if len(lane.Each) == 0 {
			if current(lane.Name, "", testKey("", lane.Name)) {
				fmt.Fprintf(output, "%s: passed, current at %.12s\n", lane.Name, commit)
			} else {
				fmt.Fprintf(output, "%s: no pass current at %.12s\n", lane.Name, commit)
				other = append(other, lane.Name)
			}
			continue
		}
		passed, not := 0, 0
		for key := range results {
			name, test, _ := strings.Cut(key, " ")
			for _, arguments := range lane.Each {
				if pkg := arguments[len(arguments)-1]; modulePrefix+strings.TrimPrefix(pkg, "./") == name {
					if current(lane.Name, pkg, testKey(pkg, test)) {
						passed++
					} else {
						not++
					}
				}
			}
		}
		fmt.Fprintf(output, "%s: %d tests kept passed and current at %.12s, %d not\n", lane.Name, passed, commit, not)
		if not > 0 || passed == 0 {
			other = append(other, lane.Name)
		}
	}
	if len(other) > 0 {
		return fmt.Errorf("-lanes-passed-at %.12s: %d lanes hold results that are not passes current there: %s", commit, len(other), strings.Join(other, ", "))
	}
	return nil
}

// testsPassedSince answers, from the results the lanes and the landings keep (plan.RecordTestResults), whether each test
// named <package>.<Test> last passed at commit or at one descending from it (since): it says each test's last result
// on output, and is an error naming the tests with no pass since, so a step whose evidence is the slow tests' asks the
// tool that keeps their records (passes-query). flash-bound-read/read's check had been an awk over the lanes' log, which
// the gate's candidate checkout does not hold and whose lines it read wrongly three times (SQA-453).
func testsPassedSince(output io.Writer, results map[string]plan.TestResult, commit string, tests []string, since func(earlier, later string) bool) error {
	if commit == "" || len(tests) == 0 {
		return errors.New("usage: validate -passed-since <commit> <package>.<Test>...")
	}
	var missing []string
	for _, named := range tests {
		at := strings.LastIndex(named, ".")
		if at <= 0 || !strings.HasPrefix(named[at+1:], "Test") {
			return fmt.Errorf("-passed-since: %q names no test as <package>.<Test>", named)
		}
		result, held := results[plan.PassedTest(named[:at], named[at+1:])]
		switch {
		case !held:
			fmt.Fprintf(output, "%s: no result kept\n", named)
			missing = append(missing, named)
		case result.Failed:
			fmt.Fprintf(output, "%s: failed at %.12s on %s\n", named, result.Commit, result.Ended.Format("2006-01-02 15:04"))
			missing = append(missing, named)
		case !since(commit, result.Commit):
			fmt.Fprintf(output, "%s: last passed at %.12s on %s, before %.12s\n", named, result.Commit, result.Ended.Format("2006-01-02 15:04"), commit)
			missing = append(missing, named)
		default:
			fmt.Fprintf(output, "%s: passed at %.12s on %s\n", named, result.Commit, result.Ended.Format("2006-01-02 15:04"))
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("-passed-since %.12s: no pass kept at it or after it for %s", commit, strings.Join(missing, ", "))
	}
	return nil
}
