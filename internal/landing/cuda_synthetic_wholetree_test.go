//go:build wholetree

package landing

import (
	"bytes"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheCUDALaneRunsNoRealModelTest holds the cuda lane to the suites on synthetic inputs it is for: no test file of
// its packages, as this host builds them, that calls the card reaches the data root, where the real models, their
// checkpoints and the store lie, through the declarations of its package it names. Such a test is the cycle's, behind
// modeltest, which the models lane runs every cycle (cuda-synthetic: the walk took in ten packages whose device tests
// opened real models, and the lane ran 35 m 54 s over them where it had run 10 m 36 s). And a test moved there is
// run: each package whose test files behind the cycle's tags call the card is one a cycle lane runs whole, or its
// moved tests would run nowhere.
func TestTheCUDALaneRunsNoRealModelTest(t *testing.T) {
	t.Parallel()
	repo := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	for _, pattern := range testcheck.Must(cudaLanePackages(repo))(t) {
		directory := filepath.Join(repo, filepath.FromSlash(strings.TrimPrefix(pattern, "./")))
		reads := dataRootReaders(t, directory)
		delete(reads, measuredInPlace[pattern])
		for _, file := range slices.Sorted(maps.Keys(reads)) {
			t.Errorf("%s/%s calls the card and reads the data root in %s; it is the cycle's, behind //go:build modeltest, which the models lane runs",
				pattern, file, strings.Join(reads[file], ", "))
		}
	}
	tagged := build.Default
	tagged.BuildTags = strings.Split(CycleTags, ",")
	run := map[string]bool{}
	for _, candidate := range cycleLanes {
		if arguments := candidate.arguments(""); !slices.Contains(arguments, "-run") {
			for _, argument := range arguments {
				run[argument] = true
			}
		}
	}
	for _, top := range []string{"internal", "cmd"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(repo, top), func(name string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(name, "_test.go") {
				return err
			}
			directory, file := filepath.Split(name)
			pattern := "./" + filepath.ToSlash(testcheck.Must(filepath.Rel(repo, filepath.Clean(directory)))(t))
			if run[pattern] || testcheck.Must(build.Default.MatchFile(directory, file))(t) || !testcheck.Must(tagged.MatchFile(directory, file))(t) {
				return nil
			}
			if bytes.Contains(testcheck.Must(os.ReadFile(name))(t), []byte(cudaRequirement)) {
				t.Errorf("%s/%s calls the card behind %s, and no cycle lane runs %s whole; its tests run nowhere", pattern, file, CycleTags, pattern)
			}
			return nil
		}))
	}
}

// measuredInPlace is, by package, the one device test file that reads the data root and stays in the gate: SenseNova's
// request resources, whose bytes are its retained measurement's identity in docs/image_video_capacity.json, so it carries
// the tag only when it is measured again on a card to itself (cuda-resources).
var measuredInPlace = map[string]string{"./internal/sensenovarecipe": "runtime_resources_cuda_windows_test.go"}

// dataRootReaders are, for each test file of directory as this host builds it that calls the card, its declarations
// that reach the data root: one that calls what the data root's packages declare (dataroot, modeltest) or the model
// directory testutil finds there, or names a declaration of the package's tests that does, followed to a fixpoint. A
// method is known by its name alone, so a call of one that reads reads. Production code is not followed: a command or
// a store finds its default under the data root, and a test that gives it a path of its own reads none
// (TestTrainRoutesReachTheMediaTrainers trains a tiny model it writes into a store of its own).
func dataRootReaders(t *testing.T, directory string) map[string][]string {
	t.Helper()
	type declaration struct {
		file, name string
		body       ast.Node
		imports    map[string]string
	}
	var declarations []declaration
	device := map[string]bool{}
	for _, entry := range testcheck.Must(os.ReadDir(directory))(t) {
		name := entry.Name()
		if !strings.HasSuffix(name, "_test.go") || !testcheck.Must(build.Default.MatchFile(directory, name))(t) {
			continue
		}
		source := testcheck.Must(os.ReadFile(filepath.Join(directory, name)))(t)
		parsed := testcheck.Must(parser.ParseFile(token.NewFileSet(), name, source, parser.SkipObjectResolution))(t)
		device[name] = bytes.Contains(source, []byte(cudaRequirement))
		imports := map[string]string{}
		for _, spec := range parsed.Imports {
			imported := testcheck.Must(strconv.Unquote(spec.Path.Value))(t)
			local := path.Base(imported)
			if spec.Name != nil {
				local = spec.Name.Name
			}
			imports[local] = imported
		}
		for _, declared := range parsed.Decls {
			switch declared := declared.(type) {
			case *ast.FuncDecl:
				if declared.Body != nil {
					declarations = append(declarations, declaration{name, declared.Name.Name, declared.Body, imports})
				}
			case *ast.GenDecl:
				for _, spec := range declared.Specs {
					if value, ok := spec.(*ast.ValueSpec); ok {
						for _, named := range value.Names {
							declarations = append(declarations, declaration{name, named.Name, value, imports})
						}
					}
				}
			}
		}
	}
	reading := map[string]bool{}
	reaches := func(candidate declaration) bool {
		found := false
		ast.Inspect(candidate.body, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.SelectorExpr:
				if owner, ok := node.X.(*ast.Ident); ok {
					switch candidate.imports[owner.Name] {
					case "overgo/internal/dataroot", "overgo/internal/modeltest":
						found = true
					case "overgo/internal/testutil":
						found = found || node.Sel.Name == "ModelArtifactDir"
					}
				}
			case *ast.Ident:
				found = found || reading[node.Name]
			}
			return !found
		})
		return found
	}
	for grew := true; grew; {
		grew = false
		for _, candidate := range declarations {
			if !reading[candidate.name] && reaches(candidate) {
				reading[candidate.name], grew = true, true
			}
		}
	}
	reads := map[string][]string{}
	for _, candidate := range declarations {
		if device[candidate.file] && reading[candidate.name] {
			reads[candidate.file] = append(reads[candidate.file], candidate.name)
		}
	}
	return reads
}
