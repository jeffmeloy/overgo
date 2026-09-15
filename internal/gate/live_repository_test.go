package gate

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/codemanifest"
	"overgo/internal/repoanalysis"
)

// liveRepository is the package's one checkout of the live repository: a
// detached worktree at HEAD with the working tree's dirty paths committed
// on top, whose package graph and manifest cache every live-tree acceptance
// shares. A plan seeds its planned paths inside the checkout and restores
// them, so an acceptance sees the same seeded candidate on a clean tree as
// in a gate candidate and never scans the live tree itself.
type liveRepository struct {
	root      string
	temporary string
	worktree  string
	graph     packageInputGraph
	manifests *codemanifest.Cache
	mutex     sync.Mutex
	plans     atomic.Int64
}

var (
	liveRepositoryOnce = sync.OnceValues(newLiveRepository)
	liveStarted        atomic.Bool
)

// liveRepositoryFixture builds the checkout on first use; TestMain removes it.
func liveRepositoryFixture(t testing.TB) *liveRepository {
	t.Helper()
	liveStarted.Store(true)
	live, err := liveRepositoryOnce()
	if err != nil {
		t.Fatal(err)
	}
	return live
}

func newLiveRepository() (*liveRepository, error) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		return nil, err
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-live-*")
	if err != nil {
		return nil, err
	}
	repository := &liveRepository{root: root, temporary: temporary, worktree: filepath.Join(temporary, "live")}
	if _, err := gitWriterCommand(root, "worktree", "add", "--quiet", "--detach", repository.worktree, "HEAD"); err != nil {
		return nil, errors.Join(err, os.RemoveAll(temporary))
	}
	if err := repository.adoptWorkingTree(); err != nil {
		return nil, errors.Join(err, repository.teardown())
	}
	repository.graph, err = loadPackageInputGraph(repository.worktree)
	if err != nil {
		return nil, errors.Join(err, repository.teardown())
	}
	repository.manifests, err = codemanifest.NewCache(8)
	if err != nil {
		return nil, errors.Join(err, repository.teardown())
	}
	// The unseeded checkout generates the base manifest every plan reuses.
	if _, _, _, err := repository.context().deriveManifestImpact(); err != nil {
		return nil, errors.Join(err, repository.teardown())
	}
	return repository, nil
}

// adoptWorkingTree commits the live working tree's dirty paths into the
// checkout, so the fixture measures the candidate the way a gate candidate
// does: HEAD plus the uncommitted change, with seeds applied on top.
func (r *liveRepository) adoptWorkingTree() error {
	raw, err := command(r.root, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return err
	}
	dirty, err := repoanalysis.ParseDirtyStatus([]byte(raw))
	if err != nil {
		return err
	}
	changed := false
	for _, entry := range dirty {
		for _, path := range []string{entry.OriginalPath, entry.Path} {
			if path == "" {
				continue
			}
			target := filepath.Join(r.worktree, filepath.FromSlash(path))
			data, err := os.ReadFile(filepath.Join(r.root, filepath.FromSlash(path)))
			switch {
			case errors.Is(err, os.ErrNotExist):
				if err := os.Remove(target); err != nil && !errors.Is(err, os.ErrNotExist) {
					return err
				}
			case err != nil:
				return err
			default:
				if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
					return err
				}
				if err := os.WriteFile(target, data, 0o644); err != nil {
					return err
				}
			}
			changed = true
		}
	}
	if !changed {
		return nil
	}
	if _, err := gitWriterCommand(r.worktree, "add", "-A"); err != nil {
		return err
	}
	_, err = gitWriterCommand(r.worktree, "-c", "user.name=overgo gate fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "live working tree")
	return err
}

func (r *liveRepository) teardown() error {
	if r == nil {
		return nil
	}
	_, err := gitWriterCommand(r.root, "worktree", "remove", "--force", r.worktree)
	return errors.Join(err, os.RemoveAll(r.temporary))
}

// context binds a gate context to the checkout as both repository and
// candidate, with the shared graph and manifest cache.
func (r *liveRepository) context(paths ...string) *gateContext {
	graph := r.graph.clone()
	return &gateContext{repo: r.worktree, paths: paths, candidateRoot: r.worktree, packageGraph: &graph, manifestCache: r.manifests}
}

// use runs one read of the unseeded checkout under the fixture lock.
func (r *liveRepository) use(t testing.TB, body func(g *gateContext)) {
	t.Helper()
	r.mutex.Lock()
	defer r.mutex.Unlock()
	body(r.context("internal/gate/preflight.go"))
}

// plan seeds every planned path, binds the seeded tree as the candidate and
// runs the acceptance; a Go seed adds one function so the manifest holds a
// structural change, a document seed changes the file's content identity.
func (r *liveRepository) plan(t testing.TB, paths []string, use func(g *gateContext)) {
	t.Helper()
	r.mutex.Lock()
	defer r.mutex.Unlock()
	defer r.seed(t, paths)()
	g := r.context(paths...)
	tree, err := g.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	g.candidateTree = tree
	r.plans.Add(1)
	use(g)
}

func (r *liveRepository) seed(t testing.TB, paths []string) func() {
	t.Helper()
	originals := map[string][]byte{}
	for _, path := range paths {
		full := filepath.Join(r.worktree, filepath.FromSlash(path))
		data, err := os.ReadFile(full)
		if err != nil {
			t.Fatal(err)
		}
		originals[full] = data
		seeded := append(slices.Clone(data), '\n')
		if strings.HasSuffix(path, ".go") {
			seeded = seedFunctionBody(t, path, data)
		}
		if err := os.WriteFile(full, seeded, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return func() {
		for full, data := range originals {
			if err := os.WriteFile(full, data, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// seedFunctionBody changes the body of the file's first function, the shape
// of a real edit: an owned symbol changes and no uncovered symbol appears.
func seedFunctionBody(t testing.TB, path string, data []byte) []byte {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), path, data, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		function, ok := decl.(*ast.FuncDecl)
		if !ok || function.Body == nil {
			continue
		}
		offset := int(function.Body.Lbrace) - 1
		return slices.Concat(data[:offset+1], []byte("\n\t_ = 0"), data[offset+1:])
	}
	t.Fatalf("%s has no function body to seed", path)
	return nil
}
