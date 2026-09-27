package landing

import (
	"context"
	"fmt"
	"os"
	"path"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/devicescope"
)

// lane is one check that drives real hardware or a browser. It runs when the
// change reaches the test closure of its packages, or touches a path under
// one of its prefixes; go list, not a manifest, decides the closure.
type lane struct {
	name     string
	packages []string
	prefixes []string
	command  func(changed string) []string
}

var lanes = []lane{
	{
		// The device lane runs its scopes' tests as its own processes, not as
		// imports, so its reach is the closure of those packages.
		name: "device", packages: append(devicePackages(), "./cmd/device-lane"), prefixes: []string{"kernels/"},
		command: func(changed string) []string { return []string{"run", "./cmd/device-lane", "-paths-file", changed} },
	},
	{
		// The page's assets, browser tests, lane and page declarations.
		name: "webui", packages: []string{"./internal/server", "./internal/webuilane", "./cmd/webui-lane"},
		prefixes: []string{"internal/server/webui/", "internal/server/webui_", "cmd/webui-lane/", "internal/webuilane/",
			"internal/server/workspace_manifest.json", "internal/server/workspace_schema.json", "internal/server/library_validation.json"},
		command: func(string) []string { return []string{"run", "./cmd/webui-lane"} },
	},
	{
		// A served-model journey or the server command.
		name:     "model-journey",
		packages: []string{"./internal/modelswap", "./internal/libraryintake", "./internal/discovery", "./cmd/server"},
		prefixes: []string{"cmd/server/", "internal/server/webui_browser_firstrun", "internal/server/webui_browser_library_validation",
			"internal/server/webui_browser_preparation", "internal/server/webui_browser_store_test.go", "internal/server/webui_served_projector",
			"internal/audioparity/webui_microphone"},
		command: func(string) []string { return []string{"run", "./cmd/webui-lane", "-journeys"} },
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
// test closure, or under a prefix that triggers it.
func (l *landing) reaches(ctx context.Context, candidate lane) (bool, error) {
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

// devicePackages are the device lane's scopes as package patterns; the cuda
// tree is recursive.
func devicePackages() []string {
	var patterns []string
	for _, packagePath := range devicescope.Packages() {
		if packagePath == "internal/cuda" {
			packagePath += "/..."
		}
		patterns = append(patterns, "./"+packagePath)
	}
	return patterns
}

func tail(text string) string {
	return clioptions.Tail(text, clioptions.DiagnosticTailBytes)
}
