package landing

import (
	"context"
	"fmt"
	"path"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/runrecord"
)

// whereDeclared is a step's check run where its named test is declared (fail-early/packages-found): a check of one go
// test naming one test (-run '^TestName$') runs in the packages of root whose test files declare that test, whatever
// packages it names, and said tells which where they differ. A check holds each package it names to a passing test of its
// name, so packages typed by hand, before the work was surveyed, refused a landing that held the test; the landing finds
// them instead. Any other check runs as written. A named test that no test file declares is refused as that.
func whereDeclared(ctx context.Context, root, verifier string) (resolved, said string, err error) {
	parsed, err := runrecord.ParseVerify(verifier)
	if err != nil || parsed.Composite {
		return verifier, "", nil
	}
	tests := parsed.GoTests()
	if len(tests) != 1 || tests[0].Target == nil {
		return verifier, "", nil
	}
	name, single := singleTestName(tests[0].Target.Pattern)
	if !single {
		return verifier, "", nil
	}
	found, err := gitauthority.Query(ctx, root, "grep", "--untracked", "-l", "-F", "-e", "func "+name+"(", "--", "*_test.go")
	if err != nil && len(found) == 0 {
		return "", "", fmt.Errorf("land: no test file of the change declares %s, which the step's check %q runs", name, verifier)
	}
	var packages []string
	for file := range strings.Lines(string(found)) {
		if file = strings.TrimSpace(file); file != "" {
			packages = append(packages, "./"+path.Dir(file))
		}
	}
	slices.Sort(packages)
	packages = slices.Compact(packages)
	named := slices.Sorted(slices.Values(tests[0].Packages))
	if slices.Equal(packages, named) {
		return verifier, "", nil
	}
	resolved, err = parsed.InPackages(verifier, packages)
	if err != nil {
		return verifier, "", nil
	}
	return resolved, fmt.Sprintf("land: the check runs %s where it is declared, %s, not %s as it names", name,
		strings.Join(packages, " "), strings.Join(tests[0].Packages, " ")), nil
}

// singleTestName is the test a selector names when it names exactly one, as ^TestName$.
func singleTestName(pattern string) (string, bool) {
	name, anchored := strings.CutPrefix(pattern, "^")
	name, closed := strings.CutSuffix(name, "$")
	if !anchored || !closed || name == "" {
		return "", false
	}
	for _, r := range name {
		if r != '_' && (r < 'A' || r > 'Z') && (r < 'a' || r > 'z') && (r < '0' || r > '9') {
			return "", false
		}
	}
	return name, true
}
