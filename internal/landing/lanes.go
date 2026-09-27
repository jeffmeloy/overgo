package landing

import (
	"context"
	"fmt"
	"os"
	"path"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/codeprofile"
)

// lane is one check that drives real hardware or a browser. It runs when the
// change reaches the test closure of its packages, or touches a path class
// that triggers it directly; go list, not a manifest, decides the closure.
type lane struct {
	name     string
	packages []string
	surface  codeprofile.Surface
	prefixes []string
	command  func(changed string) []string
}

var lanes = []lane{
	{
		name: "device", packages: []string{"./internal/cuda/...", "./cmd/device-lane"}, prefixes: []string{"kernels/"},
		command: func(changed string) []string { return []string{"run", "./cmd/device-lane", "-paths-file", changed} },
	},
	{
		name: "webui", packages: []string{"./internal/server", "./internal/webuilane", "./cmd/webui-lane"}, surface: codeprofile.WebUI,
		command: func(string) []string { return []string{"run", "./cmd/webui-lane"} },
	},
	{
		name:     "model-journey",
		packages: []string{"./internal/modelswap", "./internal/libraryintake", "./internal/discovery", "./cmd/server"},
		surface:  codeprofile.ModelJourney,
		command:  func(string) []string { return []string{"run", "./cmd/webui-lane", "-journeys"} },
	},
}

// lanes runs every lane the change reaches, one after another: they share
// the device.
func (l *landing) lanes(ctx context.Context) error {
	changed, err := os.CreateTemp("", "overgo-landing-paths-*.txt")
	if err != nil {
		return err
	}
	defer os.Remove(changed.Name())
	if _, err := changed.WriteString(strings.Join(l.paths, "\n")); err != nil {
		changed.Close()
		return err
	}
	if err := changed.Close(); err != nil {
		return err
	}
	for _, candidate := range lanes {
		reached, err := l.reaches(ctx, candidate)
		if err != nil {
			return err
		}
		if !reached {
			continue
		}
		fmt.Fprintf(l.Log, "land: lane %s runs\n", candidate.name)
		if _, err := l.run(ctx, nil, "go", candidate.command(changed.Name())...); err != nil {
			return fmt.Errorf("lane %s: %w", candidate.name, err)
		}
	}
	return nil
}

// reaches reports whether a changed path lies in a package of the lane's
// test closure, or in a class or under a prefix that triggers it.
func (l *landing) reaches(ctx context.Context, candidate lane) (bool, error) {
	if candidate.surface != 0 && codeprofile.Holds(l.paths, candidate.surface) {
		return true, nil
	}
	if slices.ContainsFunc(l.paths, func(name string) bool {
		return slices.ContainsFunc(candidate.prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) })
	}) {
		return true, nil
	}
	listed, err := l.run(ctx, nil, "go", append([]string{"list", "-e", "-deps", "-test", "-f", "{{.Dir}}"}, candidate.packages...)...)
	if err != nil {
		return false, err
	}
	closure := map[string]bool{}
	for directory := range strings.Lines(listed) {
		closure[strings.ReplaceAll(strings.TrimSpace(directory), "\\", "/")] = true
	}
	root := strings.ReplaceAll(l.Repo, "\\", "/")
	for _, name := range l.paths {
		if closure[path.Join(root, path.Dir(name))] {
			return true, nil
		}
	}
	return false, nil
}

func tail(text string) string {
	return clioptions.Tail(text, clioptions.DiagnosticTailBytes)
}
