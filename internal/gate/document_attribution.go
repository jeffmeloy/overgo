package gate

import (
	"fmt"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

// namedReaders lists the repository packages whose compiled sources name the
// repository path exactly, the readers an importer inherits. A package that
// only might reach it through an unnamed read is not a named reader; its
// own tests still run in scope. A package whose tests alone name the path
// reaches it for itself, which documentReach answers from declaredFiles.
func (graph packageInputGraph) namedReaders(path string) []string {
	var readers []string
	for _, node := range graph.nodes {
		if node.ForTest != "" || strings.HasSuffix(node.ImportPath, ".test") || !namesPath(node.namedProductionFiles, path) {
			continue
		}
		relative, err := filepath.Rel(graph.root, node.Dir)
		if err != nil || !filepath.IsLocal(relative) {
			continue
		}
		readers = append(readers, filepath.ToSlash(relative))
	}
	slices.Sort(readers)
	return slices.Compact(readers)
}

// attributeNamedDocuments resolves the non-Go uncertainty of each changed
// input that some package names: the change belongs to those readers, and
// a lane reaches it only by naming it or compiling a named reader. A
// document no package names keeps its uncertainty and the complete plan.
// The returned map binds each attributed path to its readers.
func attributeNamedDocuments(impact codemanifest.Impact, graph packageInputGraph) (codemanifest.Impact, map[string][]string, []string) {
	var notes []string
	documents := map[string][]string{}
	impact.Uncertainty = slices.DeleteFunc(slices.Clone(impact.Uncertainty), func(item codemanifest.Uncertainty) bool {
		if item.Kind != codemanifest.UncertaintyNonGo || item.Path == "" {
			return false
		}
		readers := graph.namedReaders(item.Path)
		if len(readers) == 0 && !graph.testNamed(item.Path) {
			return false
		}
		documents[item.Path] = readers
		notes = append(notes, fmt.Sprintf("document %s attributed to its named readers %s", item.Path, strings.Join(readers, ",")))
		return true
	})
	if len(documents) == 0 {
		return impact, nil, nil
	}
	return impact, documents, notes
}

// documentReach answers the dependency resolver for a changed directory that
// holds attributed documents: an owned package reaches a document when it
// names the document itself or compiles one of its named readers. The
// second result is false when the directory holds no attributed document.
func (graph packageInputGraph) documentReach(ownership automationcheck.Ownership, directory string, documents map[string][]string, relative map[string]string) (bool, bool) {
	attributed := false
	for document, readers := range documents {
		if path.Dir(document) != directory {
			continue
		}
		attributed = true
		for importPath, owned := range relative {
			if !ownership.OwnedPackage(owned) {
				continue
			}
			for _, index := range graph.byID[importPath] {
				if namesPath(graph.nodes[index].declaredFiles, document) {
					return true, true
				}
			}
			compiled, err := graph.inputNodes(importPath, false)
			if err != nil {
				return true, true
			}
			for index := range compiled {
				if dir, err := filepath.Rel(graph.root, graph.nodes[index].Dir); err == nil && slices.Contains(readers, filepath.ToSlash(dir)) {
					return true, true
				}
			}
		}
	}
	return false, attributed
}

// testNamed reports whether some package names the path in its tests alone.
func (graph packageInputGraph) testNamed(path string) bool {
	for _, node := range graph.nodes {
		if node.ForTest == "" && !strings.HasSuffix(node.ImportPath, ".test") && namesPath(node.declaredFiles, path) {
			return true
		}
	}
	return false
}
