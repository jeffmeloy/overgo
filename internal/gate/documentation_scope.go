package gate

import (
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/plan"
)

// Documentation and the live plan have gate-owned acceptance. They do not
// invalidate model experiments frozen at an earlier source/input identity.
// Keep this class narrow: nested evidence, fixtures, assets and executable
// configuration retain runtime ownership, regardless of their extension.
func documentationPath(name string) bool {
	return name == plan.Path || name == "README.md" || name == "skill.md" ||
		path.Dir(name) == "docs" && strings.EqualFold(path.Ext(name), ".md")
}

// Compiler inputs override the prose class, including embedded test data.
// Unknown runtime I/O broadens software changes; it does not turn an ordinary
// prose edit into a new model or browser validation experiment.
func documentationChanges(paths []string, graph packageInputGraph) bool {
	if len(paths) == 0 || slices.ContainsFunc(paths, func(name string) bool { return !documentationPath(name) }) {
		return false
	}
	changed := map[string]bool{}
	for _, name := range paths {
		absolute := filepath.Clean(filepath.Join(graph.root, filepath.FromSlash(name)))
		changed[absolute] = true
		for directory := range graph.sourceDirectories {
			relative, err := filepath.Rel(directory, absolute)
			if err != nil || relative != ".." && !filepath.IsAbs(relative) && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
				return false
			}
		}
	}
	for _, node := range graph.nodes {
		if slices.ContainsFunc(paths, func(name string) bool { return name != plan.Path && namesPath(node.declaredFiles, name) }) {
			return false
		}
		for _, name := range slices.Concat(node.EmbedFiles, node.TestEmbedFiles, node.XTestEmbedFiles) {
			if changed[filepath.Clean(filepath.Join(node.Dir, filepath.FromSlash(name)))] {
				return false
			}
		}
	}
	return true
}
