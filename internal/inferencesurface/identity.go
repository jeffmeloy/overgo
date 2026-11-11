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
	"strconv"
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
	// textRunner is the package of the runner a text model is evaluated
	// through. No capability of the catalog is a type of it, so it is counted
	// beside the catalog's imports where a tree holds it.
	textRunner = "internal/inference"
	// evaluationCount names, in a count, the sources that put an evaluation's
	// cases to a model (evaluationBinders).
	evaluationCount = "evaluation"
	// kernelPackage embeds the compiled kernels. An identity that reaches it
	// holds the kernel sources they are built from as well.
	kernelPackage = "internal/cuda/kernel"
	// countPackage is this package, whose stamp a build's count is linked into.
	countPackage = "internal/inferencesurface"
)

// evaluationBinders are the packages that put an evaluation's cases to a
// model: the command that opens the model and shapes what it is asked, the
// package that asks each case and keeps its answer, and the opener of a text
// model. Their own sources join the identity an evaluation's answers are kept
// under, as the catalog's own sources join a capability's. They are not
// descended into: what they reach beneath them is the answering model's
// closure, which its own identity holds, or another model's, or code that
// scores and records an answer and gives none.
var evaluationBinders = []string{"cmd/evaluate", "internal/evaluation", "internal/modelcli"}

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

// A capability's code reads its settings through internal/envsetting (setting-reader): a result setting through
// envsetting.Result, its name a quoted literal the count reads and its value joining the identity of the package whose
// code reads it, and a neutral one through envsetting.Neutral with the reason it changes nothing computed. The count
// keeps no list of either: the code it counts says them. A read of the environment through anything else leaves every
// identity that reaches it uncounted, since nothing says whether an answer depends on it.
const (
	// settingPackage is the package a setting is read through, whose own reads of the environment are those.
	settingPackage = "internal/envsetting"
	// settingQualifier and resultReader are how a read of a result setting is spelled.
	settingQualifier = "envsetting"
	resultReader     = "Result"
	// settingsMark separates, in a count, a package's digest from the names of the result settings its code reads, and
	// those from each other: none is in a digest or a setting's name.
	settingsMark = ";"
)

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
			_, read, _ := strings.Cut(counted, settingsMark)
			sum := sha256.Sum256([]byte(counted + "\n" + processEnvironment(read)))
			return hex.EncodeToString(sum[:]), true
		}
	}
	return "", false
}

// Evaluation is the identity this process keeps the answers of an evaluation
// under, for cases a model of a type of pkg answers: the package's execution
// identity with the count of the sources that put a case to the model
// (evaluationBinders). held is false when the build counted either of them
// not, and the caller then keeps the answers no longer than it would have.
func Evaluation(pkg string) (identity string, held bool) {
	model, counted := Execution(pkg)
	binders, bound := Execution(evaluationCount)
	if !counted || !bound {
		return "", false
	}
	sum := sha256.Sum256([]byte(model + "\n" + binders))
	return hex.EncodeToString(sum[:]), true
}

// processEnvironment is this process's environment under the result settings a count names, read (settingsMark
// between them): the value of each, and the libraries, read once, as a library a process has loaded stays the one it
// computes with.
func processEnvironment(read string) string {
	return settingValues(os.Getenv, settingNames(read)) + processLibraries()
}

// processLibraries are this process's libraries, read once.
var processLibraries = sync.OnceValue(func() string {
	executable, _ := os.Executable()
	roots, _ := dataroot.ResolveCurrent()
	return libraryFacts(os.Getenv, executable, roots.Libraries())
})

// settingNames are the names of the result settings a count names, read, none when it names none.
func settingNames(read string) []string {
	if read == "" {
		return nil
	}
	return strings.Split(read, settingsMark)
}

// settingValues are the value of each setting names holds, as lookup gives it.
func settingValues(lookup func(string) string, names []string) string {
	var facts strings.Builder
	for _, name := range names {
		fmt.Fprintf(&facts, "setting %s=%s\n", name, lookup(name))
	}
	return facts.String()
}

// libraryFacts describes each library as a change to it shows, its place, size and time, at every place a loader may
// take it from: beside the executable, the system directory, the search path, CUDA_PATH's bin, and beside, the
// directory of libraries beside overgo's data.
func libraryFacts(lookup func(string) string, executable, beside string) string {
	var facts strings.Builder
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
	// A tree holds the text runner and an evaluation's binders or it does not,
	// and one without them counts nothing for them.
	roots := []string{"./" + capabilityCatalog, "./" + capabilityExecutor}
	for _, held := range append([]string{textRunner}, evaluationBinders...) {
		if _, err := os.Stat(filepath.Join(root, held)); err == nil {
			roots = append(roots, "./"+held)
		}
	}
	listed, module, err := list(ctx, root, roots...)
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
	for _, entry := range append(slices.Clone(catalog.Imports), catalog.ImportPath, module+"/"+textRunner) {
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
			counted[entry] = read.counted(identity, files)
		}
	}
	var binders []string
	for _, binder := range evaluationBinders {
		pkg, found := listed[module+"/"+binder]
		if !found {
			return counted, refused, nil
		}
		own, err := packageFiles(pkg)
		if err != nil {
			return nil, nil, err
		}
		binders = append(binders, own...)
	}
	identity, undeclared, err := read.digest(root, binders, nil, "")
	if err != nil {
		return nil, nil, err
	}
	if len(undeclared) != 0 {
		refused[evaluationCount] = undeclared
	} else {
		counted[evaluationCount] = read.counted(identity, binders)
	}
	return counted, refused, nil
}

// sourceFacts is what a count reads of one file: the digest of its content
// and, of a Go source, each read of the environment nothing declares.
type sourceFacts struct {
	digest     []byte
	undeclared []string
	// results are the names of the result settings it reads (envsetting.Result).
	results []string
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
		results, other := environmentReads(file, content)
		facts.results = results
		// The package a setting is read through reads the environment for each read through it.
		if !strings.Contains(filepath.ToSlash(file), "/"+settingPackage+"/") {
			for _, spelling := range other {
				facts.undeclared = append(facts.undeclared, file+" reads "+spelling)
			}
		}
	}
	s[file] = facts
	return facts, nil
}

// counted is a package's count: the digest of its files, then the names of the result settings they read, sorted,
// settingsMark between each. The files are read already (digest).
func (s sources) counted(digest string, files []string) string {
	var names []string
	for _, file := range files {
		names = append(names, s[file].results...)
	}
	slices.Sort(names)
	return strings.Join(append([]string{digest}, slices.Compact(names)...), settingsMark)
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

// environmentReads lists what a Go source reads of the environment: the names it reads through envsetting.Result, each
// written as one quoted literal, and every other read, through Getenv or LookupEnv, or through Result with a name no
// count can read, as its argument is spelled up to the first closing parenthesis.
func environmentReads(name string, source []byte) (results, other []string) {
	resultCall := settingQualifier + "." + resultReader + "("
	if !bytes.Contains(source, []byte(resultCall)) && !slices.ContainsFunc(readers, func(reader string) bool { return bytes.Contains(source, []byte(reader+"(")) }) {
		return nil, nil
	}
	var scan scanner.Scanner
	scan.Init(token.NewFileSet().AddFile(name, -1, len(source)), source, nil, 0)
	var argument strings.Builder
	var parts int
	var quoted bool
	// before and last are the two tokens before this one, spelled, which say whether a Result is envsetting's.
	var before, last string
	called, viaResult, reading := false, false, false
	for _, kind, literal := scan.Scan(); kind != token.EOF; _, kind, literal = scan.Scan() {
		spelled := cmp.Or(literal, kind.String())
		switch {
		case reading && kind == token.RPAREN:
			reading = false
			read := argument.String()
			setting, err := strconv.Unquote(read)
			switch {
			case !viaResult:
				other = append(other, read)
			case parts == 1 && quoted && err == nil:
				results = append(results, setting)
			default:
				other = append(other, resultCall+read+")")
			}
		case reading:
			argument.WriteString(spelled)
			parts++
			quoted = kind == token.STRING
		case called && kind == token.LPAREN:
			reading, parts = true, 0
			argument.Reset()
		}
		if !reading {
			viaResult = kind == token.IDENT && literal == resultReader && last == token.PERIOD.String() && before == settingQualifier
			called = kind == token.IDENT && (slices.Contains(readers, literal) || viaResult)
		}
		before, last = last, spelled
	}
	return results, other
}
