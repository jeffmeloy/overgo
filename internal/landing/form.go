package landing

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/runrecord"
)

// form reads each landed step's check before any long stage, in seconds whether the check runs on the card or not
// (cheap-first/form): each go test it names is listed, not run (go test -list), in the change, where every package it
// names must hold a test it runs; and, where the undo run will restore the change's production Go, in that tree too,
// where a check whose tests do not build is known to give no undo evidence. A step none of whose checks builds there
// is refused now unless its note says why, and the undo run does not run such a check again.
func (l *landing) form(ctx context.Context) error {
	for _, check := range l.verifiers {
		if err := l.list(ctx, l.stageRoot(), check); err != nil {
			return fmt.Errorf("land: the check %q cannot run as written: %w", check, err)
		}
	}
	l.unbuilt = map[string]string{}
	restored, checks, none := l.undoable()
	if none != "" {
		return nil
	}
	tree, said, release, err := l.undoTree(ctx, restored)
	if err != nil {
		return err
	}
	defer release()
	for _, check := range checks {
		if err := l.list(ctx, tree, check); err != nil {
			l.unbuilt[check], _, _ = strings.Cut(err.Error(), "\n")
		}
	}
	if len(l.unbuilt) == len(checks) && len(l.boundaryReasons) == 0 {
		return l.noBoundary(said, slices.Sorted(maps.Keys(l.unbuilt)))
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
