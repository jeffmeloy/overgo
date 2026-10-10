package inferencesurface

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"

	"overgo/internal/cuda/cudnn"
	"overgo/internal/dataroot"
	"overgo/internal/processcontrol"
)

// The packages an execution identity is counted from. A capability's model is
// a type of a package the catalog imports, so each of those packages has an
// identity: its own dependency closure and the executor's, which every
// capability runs through, with the catalog's own sources, which bind each
// model to its loader. The catalog's closure is every capability's at once,
// so it is not descended into: a change to one of its own files moves every
// identity, and a change beneath it moves the identities that reach it.
const (
	capabilityCatalog  = "internal/mediacapability"
	capabilityExecutor = "internal/capabilityruntime"
	// kernelPackage embeds the compiled kernels. An identity that reaches it
	// holds the kernel sources they are built from as well.
	kernelPackage = "internal/cuda/kernel"
	// countPackage is this package, whose stamp a build's count is linked into.
	countPackage = "internal/inferencesurface"
)

// stamp is what the build counted: import/path=identity for each package,
// separated by commas. The linker sets it from the flag LinkFlag returns. A
// plain go run, go test or go build leaves it empty, and a process of such a
// build holds no identity to keep an answer under.
var stamp string

// linkFlag is the linker flag that sets the stamp of the count package at
// path to count.
func linkFlag(path, count string) string { return "-X=" + path + ".stamp=" + count }

// Linked returns the linker flag this executable's own count was recorded
// with, or nothing when it holds none: a command passes it to a command it
// starts with go run from the tree it was built from itself, whose count is
// therefore the same. A test is given none, since a test's build keeps no
// answer beyond its process.
func Linked() string {
	if stamp == "" {
		return ""
	}
	return linkFlag(reflect.TypeFor[sources]().PkgPath(), stamp)
}

// buildSettings are the settings of the Go tool a compiled result can depend
// on: the target, the instruction levels the compiler may use, which decide
// whether a multiply and an add are fused, what is compiled in, the
// toolchain, and the flags every build is given.
var buildSettings = []string{"GOOS", "GOARCH", "GOAMD64", "GOARM64", "CGO_ENABLED", "GOEXPERIMENT", "GOVERSION", "GOFLAGS"}

// resultSettings are the names in a process's environment that a capability's
// result can depend on; the value of each joins the identity of the process.
// CUDA_PATH places the cuBLAS libraries when the search path does not hold
// them, CUDA_VISIBLE_DEVICES decides which device the driver gives first, and
// ProgramFiles places the FFmpeg a clip is decoded with when nothing else
// names one.
var resultSettings = []string{"CUDA_PATH", "CUDA_VISIBLE_DEVICES", "ProgramFiles"}

// neutralReads are the other reads of the environment in the code of an
// identity, each as its argument is spelled, none of which changes what a
// capability computes: four print a diagnostic, and two pass a campaign's
// session and its resource bounds to a child process. Two more say where
// overgo's data is (internal/dataroot), which is where cuDNN is looked for
// when no search path holds it: what that changes is the library found, and
// the place and the files found there are counted with the libraries below.
// A read that is neither here nor a result setting leaves every identity that
// reaches it uncounted, since nothing says whether an answer depends on it.
var neutralReads = []string{
	`"OVERGO_LEDGER_TRACE"`, `"OVERGO_MEMORY_PLAN"`, `"OVERGO_OP_COUNTS"`, `"OVERGO_VAE_PROFILE"`,
	"CampaignEnvironment", "inheritedResourcesEnvironment", "Env", "boundEnvironment",
}

// readers are the functions a setting is read from the environment with.
var readers = []string{"Getenv", "LookupEnv"}

// libraries are the files a process loads from outside its executable that a
// device result can depend on: the CUDA driver, the cuBLAS libraries, and
// cuDNN with the parts of it a convolution loads from beside it
// (internal/cuda/driver, internal/cuda/cublas, internal/cuda/cudnn). cuDNN is
// one a machine may be without: a clip convolved through it and one convolved
// by the kernels differ, and so do the identities they are kept under.
var libraries = append([]string{"nvcuda.dll", "cublas64_12.dll", "cublasLt64_12.dll"}, cudnn.Files...)

// LinkFlag counts the execution identities of the tree at root and returns
// the linker flag that records them in an executable built from it. The
// repository's release build and the launch of a measurement pass it in
// -ldflags, so their processes keep an answer under the identity of the code
// that gave it; a build without it keeps none.
func LinkFlag(ctx context.Context, root string) (string, error) {
	counted, _, err := identities(ctx, root)
	if err != nil {
		return "", err
	}
	module, err := modulePath(root)
	if err != nil {
		return "", err
	}
	entries := make([]string, 0, len(counted))
	for pkg, identity := range counted {
		entries = append(entries, pkg+"="+identity)
	}
	slices.Sort(entries)
	return linkFlag(module+"/"+countPackage, strings.Join(entries, ",")), nil
}

// Execution is the identity this process keeps the answers of a capability
// under, for a capability whose model is a type of pkg: what the build
// counted for the package, with the settings and the libraries the process
// computes under. held is false when the build counted nothing for it, and
// the caller then keeps no answer longer than it would have without one.
func Execution(pkg string) (identity string, held bool) {
	for entry := range strings.SplitSeq(stamp, ",") {
		if counted, found := strings.CutPrefix(entry, pkg+"="); found && pkg != "" {
			sum := sha256.Sum256([]byte(counted + "\n" + processEnvironment()))
			return hex.EncodeToString(sum[:]), true
		}
	}
	return "", false
}

// processEnvironment reads this process's environment once: a library a
// process has loaded stays the one it computes with.
var processEnvironment = sync.OnceValue(func() string {
	executable, _ := os.Executable()
	roots, _ := dataroot.ResolveCurrent()
	return environment(os.Getenv, executable, roots.Libraries())
})

// environment describes what a result depends on that no build can count, as
// lookup and the disk give it: the value of each result setting, and each
// library as a change to it shows, its place, size and time, at every place a
// loader may take it from: beside the executable, the system directory, the
// search path, CUDA_PATH's bin, and beside, the directory of libraries beside
// overgo's data.
func environment(lookup func(string) string, executable, beside string) string {
	var facts strings.Builder
	for _, name := range resultSettings {
		fmt.Fprintf(&facts, "setting %s=%s\n", name, lookup(name))
	}
	directories := append([]string{filepath.Dir(executable), filepath.Join(lookup("SystemRoot"), "System32")}, filepath.SplitList(lookup("PATH"))...)
	directories = append(directories, filepath.Join(lookup("CUDA_PATH"), "bin"), beside)
	for _, library := range libraries {
		for _, directory := range directories {
			path := filepath.Join(directory, library)
			if info, err := os.Stat(path); err == nil && !info.IsDir() {
				fmt.Fprintf(&facts, "library %s %s %d %d\n", library, path, info.Size(), info.ModTime().UnixNano())
			}
		}
	}
	return facts.String()
}

// identities counts the tree at root: for each package of the module the
// capability catalog imports, and the catalog itself, the digest of every
// file a build of its closure and the executor's reads, the catalog's own
// sources, the kernel sources where the closure runs kernels, the version of
// every other module it compiles, and the build settings as the Go tool
// gives them. A package whose closure reads the environment for a name
// nothing declares is refused, with the reads, and has no identity.
func identities(ctx context.Context, root string) (counted map[string]string, refused map[string][]string, err error) {
	if root, err = filepath.Abs(root); err != nil {
		return nil, nil, err
	}
	listed, module, err := list(ctx, root, "./"+capabilityCatalog, "./"+capabilityExecutor)
	if err != nil {
		return nil, nil, err
	}
	catalog, found := listed[module+"/"+capabilityCatalog]
	if !found {
		return nil, nil, errors.New("inference surface: the tree holds no capability catalog")
	}
	var settings, failure bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: append([]string{"env"}, buildSettings...), Dir: root, Env: os.Environ(), Stdout: &settings, Stderr: &failure,
	})
	if err != nil || receipt.ExitCode != 0 {
		return nil, nil, fmt.Errorf("inference surface: go env: exit=%d: %v: %s", receipt.ExitCode, err, strings.TrimSpace(failure.String()))
	}
	binding, err := packageFiles(catalog)
	if err != nil {
		return nil, nil, err
	}
	read := sources{}
	counted, refused = map[string]string{}, map[string][]string{}
	for _, entry := range append(slices.Clone(catalog.Imports), catalog.ImportPath) {
		if pkg, found := listed[entry]; !found || pkg.Module.Path != module {
			continue
		}
		packages, modules := reach(listed, module, entry, module+"/"+capabilityExecutor)
		files := slices.Clone(binding)
		for _, pkg := range packages {
			own, err := packageFiles(pkg)
			if err != nil {
				return nil, nil, err
			}
			files = append(files, own...)
			if pkg.ImportPath != module+"/"+kernelPackage {
				continue
			}
			kernels, err := kernelFiles(root)
			if err != nil {
				return nil, nil, err
			}
			files = append(files, kernels...)
		}
		slices.Sort(files)
		identity, undeclared, err := read.digest(root, slices.Compact(files), modules, settings.String())
		if err != nil {
			return nil, nil, err
		}
		if len(undeclared) != 0 {
			refused[entry] = undeclared
		} else {
			counted[entry] = identity
		}
	}
	return counted, refused, nil
}

// sourceFacts is what a count reads of one file: the digest of its content
// and, of a Go source, each read of the environment nothing declares.
type sourceFacts struct {
	digest     []byte
	undeclared []string
}

// sources holds the files one count has read, each read once.
type sources map[string]sourceFacts

func (s sources) of(file string) (sourceFacts, error) {
	if facts, read := s[file]; read {
		return facts, nil
	}
	content, err := os.ReadFile(file)
	if err != nil {
		return sourceFacts{}, err
	}
	sum := sha256.Sum256(content)
	facts := sourceFacts{digest: sum[:]}
	if strings.HasSuffix(file, ".go") {
		for _, spelling := range environmentReads(file, content) {
			if name, quoted := strings.CutPrefix(spelling, `"`); !slices.Contains(neutralReads, spelling) &&
				!(quoted && slices.Contains(resultSettings, strings.TrimSuffix(name, `"`))) {
				facts.undeclared = append(facts.undeclared, file+" reads "+spelling)
			}
		}
	}
	s[file] = facts
	return facts, nil
}

// digest hashes the files in path order, each as its root-relative slash path
// and its content digest, then the module@version of each other module and
// the build settings when it is given any, so the digest is the same on every
// host that holds the same sources, resolves the same modules and builds
// under the same settings. It returns with it every read of the environment
// in the files that nothing declares.
func (s sources) digest(root string, files, modules []string, settings string) (digest string, undeclared []string, err error) {
	slices.Sort(files)
	surface := sha256.New()
	for _, file := range files {
		relative, err := filepath.Rel(root, file)
		if err != nil {
			return "", nil, err
		}
		facts, err := s.of(file)
		if err != nil {
			return "", nil, err
		}
		undeclared = append(undeclared, facts.undeclared...)
		fmt.Fprintf(surface, "%s\n%x\n", filepath.ToSlash(relative), facts.digest)
	}
	for _, module := range modules {
		fmt.Fprintf(surface, "module %s\n", module)
	}
	if settings != "" {
		fmt.Fprintf(surface, "settings %s\n", settings)
	}
	return hex.EncodeToString(surface.Sum(nil)), undeclared, nil
}

// environmentReads lists what a Go source asks the environment for, each as
// the argument it is spelled with up to the first closing parenthesis: a
// quoted name, or the constant that holds one.
func environmentReads(name string, source []byte) []string {
	if !slices.ContainsFunc(readers, func(reader string) bool { return bytes.Contains(source, []byte(reader+"(")) }) {
		return nil
	}
	var scan scanner.Scanner
	scan.Init(token.NewFileSet().AddFile(name, -1, len(source)), source, nil, 0)
	var reads []string
	var argument strings.Builder
	called, reading := false, false
	for _, kind, literal := scan.Scan(); kind != token.EOF; _, kind, literal = scan.Scan() {
		switch {
		case reading && kind == token.RPAREN:
			reading = false
			reads = append(reads, argument.String())
		case reading:
			argument.WriteString(cmp.Or(literal, kind.String()))
		case called && kind == token.LPAREN:
			reading = true
			argument.Reset()
		}
		called = !reading && kind == token.IDENT && slices.Contains(readers, literal)
	}
	return reads
}
