package landing

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/runrecord"
)

// form reads each landed step's check before any long stage, in seconds whether the check runs on the card or not
// (cheap-first/form): each go test it names is listed, not run (go test -list), in the change, where every package it
// names must hold a test it runs. Whether the check builds without the change is the undo run's to find and say, not
// a refusal here (the owner's review, 2026-11-14).
func (l *landing) form(ctx context.Context) error {
	for _, check := range l.verifiers {
		if err := l.list(ctx, l.stageRoot(), check); err != nil {
			// The reason comes first, the check after it: a refusal's record keeps its first line cut to causeBytes.
			return fmt.Errorf("land: a check cannot run as written: %w: %q", err, check)
		}
	}
	return nil
}

// list lists the tests each go test of check names, in directory, by the check's own environment and build flags,
// and says which package holds no test the check runs, or that a package does not build.
func (l *landing) list(ctx context.Context, directory, check string) error {
	parsed, err := runrecord.ParseVerify(check)
	if err != nil {
		return err
	}
	for _, segment := range parsed.GoTests() {
		flags := slices.DeleteFunc(slices.Clone(segment.Flags), func(flag string) bool { return flag == "-json" })
		for _, named := range segment.Packages {
			listed, err := l.runIn(ctx, directory, append(gitauthority.RepositoryEnvironment(), segment.Env...), "go",
				slices.Concat([]string{"test", "-list", "."}, flags, []string{named})...)
			if err != nil {
				return fmt.Errorf("%s does not build: %w", named, err)
			}
			if !holdsTest(listed, segment.Target) {
				return fmt.Errorf("%s holds no test it runs", named)
			}
		}
	}
	return nil
}

// holdsTest says a package's go test -list output names a test the target runs, any test where it has none: the
// lines before the package's closing "ok" line, a test's name each.
func holdsTest(listed string, target *runrecord.GoTestTarget) bool {
	for line := range strings.Lines(listed) {
		name := strings.TrimSpace(line)
		if strings.HasPrefix(name, "Test") && (target == nil || target.RunsTest(name)) {
			return true
		}
	}
	return false
}
