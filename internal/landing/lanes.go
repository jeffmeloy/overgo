package landing

import (
	"context"
	"path"
	"slices"
	"strings"

	"overgo/internal/clioptions"
)

// lane is what a check depends on: the test closure of its packages, and
// the paths under its prefixes; go list, not a manifest, decides the
// closure. The device, browser and model-journey lanes run on the
// schedule, not in the landing: they block no commit on hardware or a
// browser.
type lane struct {
	packages []string
	prefixes []string
}

// reaches reports whether a changed path lies in a package of the lane's
// test closure, or under a prefix that triggers it.
func (l *landing) reaches(ctx context.Context, candidate lane) (bool, error) {
	if slices.ContainsFunc(l.paths, func(name string) bool {
		return slices.ContainsFunc(candidate.prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) })
	}) {
		return true, nil
	}
	closure, err := l.listDirectories(ctx, append([]string{"-deps", "-test"}, candidate.packages...))
	if err != nil {
		return false, err
	}
	own, err := l.listDirectories(ctx, candidate.packages)
	if err != nil {
		return false, err
	}
	root := strings.ReplaceAll(l.Repo, "\\", "/")
	for _, name := range l.paths {
		directory := path.Join(root, path.Dir(name))
		// A test file compiles only into its own package's test binary, so
		// it reaches the lane only when that package is the lane's.
		if strings.HasSuffix(name, "_test.go") {
			if own[directory] {
				return true, nil
			}
			continue
		}
		if closure[directory] {
			return true, nil
		}
	}
	return false, nil
}

// listDirectories is the set of package directories go list names for the
// arguments, with forward slashes.
func (l *landing) listDirectories(ctx context.Context, arguments []string) (map[string]bool, error) {
	listed, err := l.run(ctx, nil, "go", append([]string{"list", "-e", "-f", "{{.Dir}}"}, arguments...)...)
	if err != nil {
		return nil, err
	}
	directories := map[string]bool{}
	for directory := range strings.Lines(listed) {
		directories[strings.ReplaceAll(strings.TrimSpace(directory), "\\", "/")] = true
	}
	return directories, nil
}

func tail(text string) string {
	return clioptions.Tail(text, clioptions.DiagnosticTailBytes)
}
