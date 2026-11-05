package repoanalysis

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"overgo/internal/processcontrol"
)

// listedPackage is one package go list gives the typed walk.
type listedPackage struct {
	ImportPath, Name, Dir, Export                string
	GoFiles, CgoFiles, TestGoFiles, XTestGoFiles []string
	Imports, TestImports, XTestImports           []string
	ImportMap                                    map[string]string
	Module                                       *struct{ Main bool }
	Error                                        *struct{ Err string }
}

// listFields are the fields of listedPackage go list fills.
const listFields = "-json=ImportPath,Name,Dir,Export,GoFiles,CgoFiles,TestGoFiles,XTestGoFiles,Imports,TestImports,XTestImports,ImportMap,Module,Error"

// exportFlags list a package with every package it links, and the export data of each.
var exportFlags = []string{"-deps", "-export"}

// goList runs go list with flags in root over patterns and returns the packages it lists, in its order.
func goList(ctx context.Context, root string, flags []string, patterns ...string) ([]listedPackage, error) {
	var listed, failure bytes.Buffer
	arguments := slices.Concat([]string{"list", listFields}, flags, patterns)
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: arguments, Dir: root, Stdout: &listed, Stderr: &failure})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("exit %d: %s", receipt.ExitCode, strings.TrimSpace(failure.String()))
	}
	if err != nil {
		return nil, fmt.Errorf("go %s: %w", strings.Join(arguments, " "), err)
	}
	var packages []listedPackage
	for decoder := json.NewDecoder(&listed); ; {
		var pkg listedPackage
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			return packages, nil
		} else if err != nil {
			return nil, err
		}
		if pkg.Error != nil {
			return nil, fmt.Errorf("go list: %s: %s", pkg.ImportPath, pkg.Error.Err)
		}
		packages = append(packages, pkg)
	}
}

// importerFunc is a types.Importer of one function.
type importerFunc func(string) (*types.Package, error)

func (f importerFunc) Import(imported string) (*types.Package, error) { return f(imported) }

// checkedPackage is one package of the module, type-checked from its source.
type checkedPackage struct {
	listed *listedPackage
	pkg    *types.Package
	info   *types.Info
	files  []*ast.File
	done   chan struct{}
	err    error
}

// typedFunction is one function or method of a package a command links.
//   - key = file + receiver + name, because build tags give one name a declaration per file
//   - at = file, line and name, which says where its declaration stands
//   - lines = its source lines, its comment among them
type typedFunction struct {
	key, at string
	lines   int
}

// typedReach is what one typed walk from the commands reached.
//   - functions = the functions and methods it read, outside generated files and test support
//   - entered holds the at of every declaration the walk entered
type typedReach struct {
	functions []typedFunction
	entered   map[string]bool
}

// typedWalk is the state of one walk of types from the commands.
//   - declared maps each declared object of the module to the declarations that state it
//   - named = the named types the walk reached, in the order it reached them, and stated = the interfaces a reached declaration calls through or states, each once
//   - settled holds each named type and interface whose methods the walk has reached, and listed counts the named types whose methods an outside interface names it has reached
type typedWalk struct {
	declared map[types.Object][]typedDeclaration
	reached  map[types.Object]bool
	queue    []types.Object
	named    []*types.Named
	stated   []*types.Interface
	met      map[*types.Interface]bool
	settled  map[typePair]bool
	listed   int
}

// typePair is a named type and an interface the walk settled.
type typePair struct {
	named  *types.Named
	stated *types.Interface
}

// state keeps an interface the walk met, once.
func (w *typedWalk) state(stated *types.Interface) {
	if !w.met[stated] {
		w.met[stated] = true
		w.stated = append(w.stated, stated)
	}
}

// typedDeclaration is one declaration of the module with the type information of its package.
type typedDeclaration struct {
	node ast.Node
	info *types.Info
}

// reach marks a declared object as reached; a method of an instance is its origin's.
func (w *typedWalk) reach(object types.Object) {
	if function, isFunction := object.(*types.Func); isFunction {
		object = function.Origin()
	}
	if _, known := w.declared[object]; known && !w.reached[object] {
		w.reached[object] = true
		w.queue = append(w.queue, object)
		if named, isNamed := object.Type().(*types.Named); isNamed && object == named.Obj() {
			w.named = append(w.named, named)
		}
	}
}

// drain enters every queued declaration and reaches what its types say it uses.
//   - an identifier reaches the object it uses, and a selector the field or method it selects, as the type checker resolved it
//   - a call through an interface needs that method of every reached type that implements the interface
//   - a stated interface needs each method it lists of every reached type it holds, called or not
func (w *typedWalk) drain() {
	for len(w.queue) > 0 {
		next := w.queue[len(w.queue)-1]
		w.queue = w.queue[:len(w.queue)-1]
		for _, declaration := range w.declared[next] {
			ast.Inspect(declaration.node, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.Ident:
					if used := declaration.info.Uses[typed]; used != nil {
						w.reach(used)
					}
				case *ast.SelectorExpr:
					if selection := declaration.info.Selections[typed]; selection != nil {
						if stated, isInterface := selection.Recv().Underlying().(*types.Interface); isInterface {
							w.state(stated)
						}
						w.reach(selection.Obj())
					}
				case *ast.InterfaceType:
					if stated, isInterface := declaration.info.TypeOf(typed).(*types.Interface); isInterface {
						w.state(stated)
					}
				}
				return true
			})
		}
	}
}

// owed returns the methods the reached named types owe: for a type or its pointer that implements an interface the walk
// met, each method of the interface as the type's method set gives it, its own or one an embedded type lends it; and
// each method of its pointer's method set that an interface outside the module names.
func (w *typedWalk) owed(outside map[string]bool) []types.Object {
	var owed []types.Object
	for ; w.listed < len(w.named); w.listed++ {
		for selection := range types.NewMethodSet(types.NewPointer(w.named[w.listed])).Methods() {
			if outside[selection.Obj().Name()] {
				owed = append(owed, selection.Obj())
			}
		}
	}
	for _, named := range w.named {
		for _, stated := range w.stated {
			if pair := (typePair{named, stated}); !w.settled[pair] {
				w.settled[pair] = true
				owed = append(owed, implementation(named, stated)...)
			}
		}
	}
	return owed
}

// implementation returns the methods by which a named type, or its pointer, implements an interface, and none when neither does.
func implementation(named *types.Named, stated *types.Interface) []types.Object {
	if stated.NumMethods() == 0 {
		return nil
	}
	first := stated.Method(0)
	for _, receiver := range []types.Type{named, types.NewPointer(named)} {
		methods := types.NewMethodSet(receiver)
		if methods.Lookup(first.Pkg(), first.Name()) == nil || !types.Implements(receiver, stated) {
			continue
		}
		var held []types.Object
		for method := range stated.Methods() {
			held = append(held, methods.Lookup(method.Pkg(), method.Name()).Obj())
		}
		return held
	}
	return nil
}

// testSupportFunction reports whether a function takes a testing.TB, a *testing.T, a *testing.B or a *testing.F, which makes it test support in whatever package declares it.
func testSupportFunction(function *types.Func) bool {
	parameters := function.Signature().Params()
	for v := range parameters.Variables() {
		parameter := v.Type()
		if pointer, isPointer := parameter.(*types.Pointer); isPointer {
			parameter = pointer.Elem()
		}
		if named, isNamed := parameter.(*types.Named); isNamed && named.Obj().Pkg() != nil && named.Obj().Pkg().Path() == "testing" {
			return true
		}
	}
	return false
}

// declarationName returns a function's name as the census keys it: receiver.name for a method.
func declarationName(function *ast.FuncDecl) string {
	if function.Recv == nil {
		return function.Name.Name
	}
	return receiverType(function) + "." + function.Name.Name
}

// reachTypes walks the module at root by types, for the host's build, from its commands.
//   - linked = the packages some command links, through the imports of their production files
//   - roots = main of every command + init, variables and constants of every linked package
//   - a lane = a command that lists packages of the module in a string slice, whose tests it runs; a declaration of a package it both links and lists that such a test names is a root, because the lane reaches it
//   - a method is reached <=> it is selected, or a reached type owes it to an interface the walk met or one outside the module names, through its method set, where an embedded type lends it
//   - outside = the method names of interfaces declared outside the module, whose callers the walk cannot read
//   - functions = the functions and methods of linked packages outside generated files, less test support
//   - everyTest = every test of the module is a root too, as every lane were: the walk then says what no command and no test reaches
func reachTypes(ctx context.Context, root string, outside map[string]bool, everyTest bool) (typedReach, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return typedReach{}, err
	}
	listed, err := goList(ctx, root, exportFlags, "./cmd/...", "./internal/...")
	if err != nil {
		return typedReach{}, err
	}
	exports, module := map[string]string{}, map[string]*checkedPackage{}
	for index := range listed {
		pkg := &listed[index]
		exports[pkg.ImportPath] = pkg.Export
		if pkg.Module != nil && pkg.Module.Main {
			module[pkg.ImportPath] = &checkedPackage{listed: pkg, done: make(chan struct{})}
		}
	}
	fset := token.NewFileSet()
	var outsideGuard sync.Mutex
	exported := importer.ForCompiler(fset, "gc", func(imported string) (io.ReadCloser, error) {
		if exports[imported] == "" {
			return nil, fmt.Errorf("no export data for %s", imported)
		}
		return os.Open(exports[imported])
	})
	importFor := func(pkg *listedPackage) types.Importer {
		return importerFunc(func(imported string) (*types.Package, error) {
			if mapped, found := pkg.ImportMap[imported]; found {
				imported = mapped
			}
			if dependency, local := module[imported]; local {
				<-dependency.done
				return dependency.pkg, dependency.err
			}
			outsideGuard.Lock()
			defer outsideGuard.Unlock()
			return exported.Import(imported)
		})
	}
	parse := func(pkg *listedPackage, names []string) ([]*ast.File, error) {
		var files []*ast.File
		for _, name := range names {
			file, err := parser.ParseFile(fset, filepath.Join(pkg.Dir, name), nil, parser.SkipObjectResolution|parser.ParseComments)
			if err != nil {
				return nil, err
			}
			files = append(files, file)
		}
		return files, nil
	}
	// check type-checks files as importPath. A test binary's files are read through their type errors: their external
	// tests import the package as its test binary builds it (tested), and what else they import links the package as
	// production builds it, which go test reconciles by building those again and the walk does not, so a check of tests
	// records what they name and reports no error.
	check := func(pkg *listedPackage, importPath string, files []*ast.File, tested *types.Package) (*types.Package, *types.Info, error) {
		info := &types.Info{Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}, Selections: map[*ast.SelectorExpr]*types.Selection{}, Types: map[ast.Expr]types.TypeAndValue{}}
		imports := importFor(pkg)
		if tested != nil {
			imports = importerFunc(func(imported string) (*types.Package, error) {
				if imported == tested.Path() {
					return tested, nil
				}
				return importFor(pkg).Import(imported)
			})
		}
		config := types.Config{Importer: imports, FakeImportC: true}
		if strings.HasSuffix(fset.Position(files[len(files)-1].Pos()).Filename, "_test.go") {
			config.Error = func(error) {}
			checked, _ := config.Check(importPath, fset, files, info)
			return checked, info, nil
		}
		checked, err := config.Check(importPath, fset, files, info)
		return checked, info, err
	}
	var group sync.WaitGroup
	for importPath, checked := range module {
		group.Go(func() {
			defer close(checked.done)
			if checked.files, checked.err = parse(checked.listed, slices.Concat(checked.listed.GoFiles, checked.listed.CgoFiles)); checked.err == nil {
				checked.pkg, checked.info, checked.err = check(checked.listed, importPath, checked.files, nil)
			}
		})
	}
	group.Wait()
	for importPath, checked := range module {
		if checked.err != nil {
			return typedReach{}, fmt.Errorf("type-check %s: %w", importPath, checked.err)
		}
	}

	// Each command's linked closure, and the packages of the module whose tests a lane runs.
	closure := func(start string) map[string]bool {
		seen := map[string]bool{}
		for queue := []string{start}; len(queue) > 0; {
			next := queue[len(queue)-1]
			queue = queue[:len(queue)-1]
			if seen[next] || module[next] == nil {
				continue
			}
			seen[next] = true
			queue = append(queue, module[next].listed.Imports...)
		}
		return seen
	}
	linked, lanes := map[string]bool{}, map[string][]string{}
	for importPath, checked := range module {
		if checked.listed.Name != "main" {
			continue
		}
		for linkedPath := range closure(importPath) {
			linked[linkedPath] = true
		}
		// The packages a command lists in a string slice are those it runs the tests of, as cmd/webui-lane lists the
		// packages of its browser tests; a package named in any other string is data.
		for _, file := range checked.files {
			ast.Inspect(file, func(node ast.Node) bool {
				listing, isLiteral := node.(*ast.CompositeLit)
				if slice, isSlice := checked.info.TypeOf(listing).(*types.Slice); !isLiteral || !isSlice || !types.Identical(slice.Elem(), types.Typ[types.String]) {
					return true
				}
				for _, element := range listing.Elts {
					if literal, isString := element.(*ast.BasicLit); isString && literal.Kind == token.STRING {
						if named, err := strconv.Unquote(literal.Value); err == nil && module[named] != nil && !slices.Contains(lanes[importPath], named) {
							lanes[importPath] = append(lanes[importPath], named)
						}
					}
				}
				return true
			})
		}
	}
	// tested maps each package whose tests are roots to the packages whose declarations they reach: a lane's own, or,
	// for every test, any of the module.
	tested := map[string]func(string) bool{}
	for command, ran := range lanes {
		// The lane's own packages are those it both links and runs, as cmd/webui-lane links internal/webuilane, whose
		// legs and waits the browser tests it runs call.
		linkedByLane := closure(command)
		for _, importPath := range ran {
			tested[importPath] = func(declared string) bool { return linkedByLane[declared] && slices.Contains(ran, declared) }
		}
	}
	if everyTest {
		// A test behind a build tag, as a real model's or the cycle's are, is a test too.
		tags, err := testTags(ctx, root, module)
		if err != nil {
			return typedReach{}, err
		}
		tagged, err := goList(ctx, root, []string{"-tags", tags}, "./cmd/...", "./internal/...")
		if err != nil {
			return typedReach{}, err
		}
		for _, pkg := range tagged {
			if checked := module[pkg.ImportPath]; checked != nil {
				checked.listed.TestGoFiles, checked.listed.XTestGoFiles = pkg.TestGoFiles, pkg.XTestGoFiles
				checked.listed.TestImports, checked.listed.XTestImports = pkg.TestImports, pkg.XTestImports
			}
		}
		for importPath := range module {
			tested[importPath] = func(declared string) bool { return module[declared] != nil }
		}
	}
	// What the tests import beyond the commands' builds, with the export data of each.
	var testImports []string
	for importPath := range tested {
		for _, imported := range slices.Concat(module[importPath].listed.TestImports, module[importPath].listed.XTestImports) {
			if _, known := exports[imported]; !known && module[imported] == nil && !slices.Contains(testImports, imported) {
				testImports = append(testImports, imported)
			}
		}
	}
	if len(testImports) > 0 {
		more, err := goList(ctx, root, exportFlags, testImports...)
		if err != nil {
			return typedReach{}, err
		}
		for _, pkg := range more {
			if _, known := exports[pkg.ImportPath]; !known {
				exports[pkg.ImportPath] = pkg.Export
			}
		}
	}

	walk := typedWalk{declared: map[types.Object][]typedDeclaration{}, reached: map[types.Object]bool{}, met: map[*types.Interface]bool{}, settled: map[typePair]bool{}}
	at := func(position token.Pos, name string) string {
		place := fset.Position(position)
		relative, _ := filepath.Rel(root, place.Filename)
		return fmt.Sprintf("%s:%d %s", filepath.ToSlash(relative), place.Line, name)
	}
	byAt := map[string]types.Object{}
	var reach typedReach
	var roots []types.Object
	// Every package of the module is walked, since a test reaches a linked package through test support too; only a
	// linked package's declarations are roots and are read.
	for importPath, checked := range module {
		for _, file := range checked.files {
			relative, _ := filepath.Rel(root, fset.Position(file.Pos()).Filename)
			relative = filepath.ToSlash(relative)
			listedFunction := linked[importPath] && !ast.IsGenerated(file)
			for _, declaration := range file.Decls {
				switch typed := declaration.(type) {
				case *ast.FuncDecl:
					function, isFunction := checked.info.Defs[typed.Name].(*types.Func)
					if !isFunction {
						continue
					}
					walk.declared[function] = append(walk.declared[function], typedDeclaration{typed, checked.info})
					byAt[at(typed.Name.Pos(), typed.Name.Name)] = function
					if typed.Recv == nil && linked[importPath] && (typed.Name.Name == "init" || checked.listed.Name == "main" && typed.Name.Name == "main") {
						roots = append(roots, function)
					}
					if listedFunction && !testSupportFunction(function) {
						from := typed.Pos()
						if typed.Doc != nil {
							from = typed.Doc.Pos()
						}
						reach.functions = append(reach.functions, typedFunction{
							key: relative + " " + declarationName(typed), at: at(typed.Name.Pos(), typed.Name.Name),
							lines: fset.Position(typed.End()).Line - fset.Position(from).Line + 1,
						})
					}
				case *ast.GenDecl:
					for _, spec := range typed.Specs {
						switch spec := spec.(type) {
						case *ast.TypeSpec:
							if object := checked.info.Defs[spec.Name]; object != nil {
								walk.declared[object] = append(walk.declared[object], typedDeclaration{spec, checked.info})
								byAt[at(spec.Name.Pos(), spec.Name.Name)] = object
							}
						case *ast.ValueSpec:
							for _, name := range spec.Names {
								if object := checked.info.Defs[name]; object != nil {
									walk.declared[object] = append(walk.declared[object], typedDeclaration{spec, checked.info})
									byAt[at(name.Pos(), name.Name)] = object
									if linked[importPath] {
										roots = append(roots, object)
									}
								}
							}
						}
					}
				}
			}
		}
	}
	// The tests that are roots: each package's, type-checked again as its test binary builds it, beside the commands.
	var testGuard sync.Mutex
	var testFailure error
	for importPath, reaches := range tested {
		group.Go(func() {
			checked := module[importPath]
			testFiles, err := parse(checked.listed, checked.listed.TestGoFiles)
			external, externalErr := parse(checked.listed, checked.listed.XTestGoFiles)
			var found []types.Object
			testBinary := checked.pkg
			for _, binary := range []struct {
				importPath string
				files      []*ast.File
			}{{importPath, append(slices.Clone(checked.files), testFiles...)}, {importPath + "_test", external}} {
				if err != nil || externalErr != nil || len(binary.files) == len(checked.files) && binary.importPath == importPath || len(binary.files) == 0 {
					continue
				}
				var tested *types.Package
				if binary.importPath != importPath {
					tested = testBinary
				}
				built, info, checkErr := check(checked.listed, binary.importPath, binary.files, tested)
				if binary.importPath == importPath {
					testBinary = built
				}
				if checkErr != nil {
					err = fmt.Errorf("type-check the tests of %s: %w", importPath, checkErr)
					break
				}
				for _, used := range info.Uses {
					if used.Pkg() != nil && reaches(used.Pkg().Path()) {
						if object := byAt[at(used.Pos(), used.Name())]; object != nil {
							found = append(found, object)
						}
					}
				}
			}
			testGuard.Lock()
			defer testGuard.Unlock()
			roots, testFailure = append(roots, found...), cmp.Or(testFailure, err, externalErr)
		})
	}
	group.Wait()
	if testFailure != nil {
		return typedReach{}, testFailure
	}
	for _, root := range roots {
		walk.reach(root)
	}
	for grew := true; grew; {
		walk.drain()
		grew = false
		for _, method := range walk.owed(outside) {
			if function, isFunction := method.(*types.Func); isFunction && !walk.reached[function.Origin()] {
				walk.reach(method)
				grew = grew || walk.reached[function.Origin()]
			}
		}
	}
	reach.entered = map[string]bool{}
	for object := range walk.reached {
		for _, declaration := range walk.declared[object] {
			if function, isFunction := declaration.node.(*ast.FuncDecl); isFunction {
				reach.entered[at(function.Name.Pos(), function.Name.Name)] = true
			}
		}
	}
	return reach, nil
}

// fixtureModule writes a module named overgo of the files given, in a directory of the test's, and returns it.
func fixtureModule(t *testing.T, files map[string]string) string {
	t.Helper()
	repo := t.TempDir()
	files = maps.Clone(files)
	files["go.mod"] = "module overgo\n\ngo 1.26\n"
	for name, content := range files {
		target := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return repo
}

// testTags returns, joined by commas, the build tags the module's test files build behind: each tag their build
// constraints name, less the systems and architectures the toolchain knows, ignoreTag and the toolchain's own.
func testTags(ctx context.Context, root string, module map[string]*checkedPackage) (string, error) {
	var known, failure bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: []string{"tool", "dist", "list"}, Dir: root, Stdout: &known, Stderr: &failure})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("exit %d: %s", receipt.ExitCode, strings.TrimSpace(failure.String()))
	}
	if err != nil {
		return "", fmt.Errorf("go tool dist list: %w", err)
	}
	platform := map[string]bool{ignoreTag: true, "cgo": true, "gc": true, "gccgo": true, "unix": true}
	for pair := range strings.FieldsSeq(known.String()) {
		system, architecture, _ := strings.Cut(pair, "/")
		platform[system], platform[architecture] = true, true
	}
	var tags []string
	for _, checked := range module {
		entries, err := os.ReadDir(checked.listed.Dir)
		if err != nil {
			return "", err
		}
		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(checked.listed.Dir, entry.Name()), nil, parser.PackageClauseOnly|parser.ParseComments)
			if err != nil {
				return "", err
			}
			for _, group := range file.Comments {
				for _, comment := range group.List {
					expression, err := constraint.Parse(comment.Text)
					if comment.Pos() > file.Package || !constraint.IsGoBuild(comment.Text) || err != nil {
						continue
					}
					expression.Eval(func(tag string) bool {
						if !platform[tag] && !strings.HasPrefix(tag, "go1.") && !slices.Contains(tags, tag) {
							tags = append(tags, tag)
						}
						return false
					})
				}
			}
		}
	}
	slices.Sort(tags)
	return strings.Join(tags, ","), nil
}

// hostTwin is the file suffix of the build-tag twin the host builds in the fixture rows.
var hostTwin = map[bool]string{true: "windows", false: "unix"}[runtime.GOOS == "windows"]

// TestTheReachWalkReadsTypes holds the walk from the commands to the types the checker resolves: a method is reached as
// the type checker selects it, so a name two types share reaches the one called alone; an interface needs the method of
// each reached type it holds; a function that takes a testing.TB is test support wherever it is declared; and a lane
// reaches what the tests it runs name. Each row pins one rule on a fixture module, built for the host.
func TestTheReachWalkReadsTypes(t *testing.T) {
	t.Parallel()
	const command = "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { lib.Use() }\n"
	for _, row := range []struct {
		rule  string
		files map[string]string
		want  []string
	}{
		{"a method a type is called by reaches that type's method alone, though another type has its name", map[string]string{
			"internal/lib/lib.go": "package lib\n\ntype Image struct{}\n\nfunc (Image) Forward() int { return 1 }\n\ntype Video struct{}\n\nfunc (Video) Forward() int { return 2 }\n\n" +
				"func Use() (int, Video) { return Image{}.Forward(), Video{} }\n",
		}, []string{"internal/lib/lib.go Video.Forward"}},
		{"the walk reads the host's build, and each twin is listed by its file", map[string]string{
			"internal/lib/lib.go":          "package lib\n\nfunc Use() int { return free() }\n",
			"internal/lib/free_windows.go": "//go:build windows\n\npackage lib\n\nfunc free() int { return 1 }\n\nfunc spare() {}\n",
			"internal/lib/free_unix.go":    "//go:build !windows\n\npackage lib\n\nfunc free() int { return 2 }\n\nfunc spare() {}\n",
		}, []string{"internal/lib/free_" + hostTwin + ".go spare"}},
		{"a package variable reaches the function it holds, called or not", map[string]string{
			"internal/lib/lib.go": "package lib\n\nvar held = map[string]func(){\"start\": start}\n\nfunc Use() {}\n\nfunc start() {}\n\nfunc idle() {}\n",
		}, []string{"internal/lib/lib.go idle"}},
		{"a method is reached through its receiver type and its selection, and an outside interface's name", map[string]string{
			"cmd/tool/main.go": "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() {\n\tvar local struct{ grow int }\n\t_, _ = local.grow, lib.Box{}.Area()\n}\n",
			"internal/lib/lib.go": "package lib\n\ntype Box struct{}\n\nfunc (Box) Area() int { return 1 }\n\nfunc (Box) Spare() int { return 2 }\n\nfunc (*Box) String() string { return \"\" }\n\nfunc (Box) grow() {}\n\n" +
				"type Lid[T any] struct{}\n\nfunc (*Lid[T]) Area() int { return 3 }\n",
		}, []string{"internal/lib/lib.go Box.Spare", "internal/lib/lib.go Box.grow", "internal/lib/lib.go Lid.Area"}},
		{"an interface reaches the method it declares of each reached type it holds, called or not, and a call through it the method it calls", map[string]string{
			"internal/lib/lib.go": "package lib\n\ntype node interface{ isNode() }\n\ntype leaf struct{}\n\nfunc (leaf) isNode() {}\n\nfunc (leaf) spare() {}\n\n" +
				"type shape interface{ Area() int }\n\ntype disc struct{}\n\nfunc (disc) Area() int { return 1 }\n\ntype square struct{}\n\nfunc (square) Area() int { return 2 }\n\n" +
				"func Use() (node, int) { var s shape = disc{}; return leaf{}, s.Area() }\n",
		}, []string{"internal/lib/lib.go leaf.spare", "internal/lib/lib.go square.Area"}},
		{"a type that implements an interface through an embedded type owes the method the embedded type lends", map[string]string{
			"internal/lib/lib.go": "package lib\n\ntype backend interface {\n\tForward() int\n\tBackward() int\n}\n\ntype forward struct{}\n\nfunc (forward) Forward() int { return 1 }\n\nfunc (forward) spare() {}\n\n" +
				"type training struct{ forward }\n\nfunc (training) Backward() int { return 2 }\n\nfunc Use() int { var b backend = training{}; return b.Forward() + b.Backward() }\n",
		}, []string{"internal/lib/lib.go forward.spare"}},
		{"a function only a test calls is listed, and one that takes a testing.TB is test support", map[string]string{
			"internal/lib/lib.go":      "package lib\n\nimport \"testing\"\n\nfunc Use() {}\n\nfunc OnlyTested() {}\n\nfunc Skip(t testing.TB) { t.Helper() }\n",
			"internal/lib/lib_test.go": "package lib\n\nimport \"testing\"\n\nfunc TestOnly(t *testing.T) { OnlyTested(); Skip(t) }\n",
		}, []string{"internal/lib/lib.go OnlyTested"}},
		{"a package no command links is test support, and what it alone calls is listed", map[string]string{
			"internal/lib/lib.go":         "package lib\n\nfunc Use() {}\n\nfunc OnlySupport() {}\n",
			"internal/libtest/libtest.go": "package libtest\n\nimport \"overgo/internal/lib\"\n\nfunc Fixture() { lib.OnlySupport() }\n\nfunc spare() {}\n",
		}, []string{"internal/lib/lib.go OnlySupport"}},
		{"an init of a package a command links reaches what it calls", map[string]string{
			"cmd/tool/main.go":    "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { println(lib.Name) }\n",
			"internal/lib/lib.go": "package lib\n\nconst Name = \"lib\"\n\nfunc init() { register() }\n\nfunc register() {}\n\nfunc spare() {}\n",
		}, []string{"internal/lib/lib.go spare"}},
		{"a generated file reaches what it calls and is not listed", map[string]string{
			"internal/lib/lib.go":          "package lib\n\nfunc Use() {}\n\nfunc fromGenerated() {}\n\nfunc spare() {}\n",
			"internal/lib/zz_generated.go": "// Code generated by fixture. DO NOT EDIT.\n\npackage lib\n\nfunc init() { fromGenerated() }\n\nfunc unused() {}\n",
		}, []string{"internal/lib/lib.go spare"}},
		{"a local that shadows a function, and a field key of its name, reach nothing", map[string]string{
			"internal/lib/lib.go": "package lib\n\ntype options struct{ limit int }\n\nfunc limit() int { return 1 }\n\nfunc scale() int { return 2 }\n\nfunc Use() options { scale := 3; return options{limit: scale} }\n",
		}, []string{"internal/lib/lib.go limit", "internal/lib/lib.go scale"}},
		{"a lane reaches what the tests of a package it lists name, in a package it both links and lists", map[string]string{
			"cmd/lane/main.go":         "package main\n\nimport \"overgo/internal/harness\"\n\nvar ran = []string{\"overgo/internal/lib\", \"overgo/internal/harness\"}\n\nfunc main() { harness.Run(ran) }\n",
			"internal/harness/run.go":  "package harness\n\nfunc Run([]string) {}\n\nfunc Leg() {}\n\nfunc Spare() {}\n",
			"internal/lib/lib.go":      "package lib\n\nfunc Use() {}\n",
			"internal/lib/lib_test.go": "package lib\n\nimport (\n\t\"testing\"\n\n\t\"overgo/internal/harness\"\n)\n\nfunc TestLeg(t *testing.T) { harness.Leg() }\n",
		}, []string{"internal/harness/run.go Spare"}},
	} {
		t.Run(row.rule, func(t *testing.T) {
			t.Parallel()
			files := map[string]string{"cmd/tool/main.go": command}
			maps.Copy(files, row.files)
			found, _, err := unreachedDeclarations(t.Context(), fixtureModule(t, files), map[string]bool{"String": true})
			var keys []string
			for _, declaration := range found {
				keys = append(keys, declaration.key)
			}
			if err != nil || !slices.Equal(keys, row.want) {
				t.Errorf("unreached = %q, %v, want %q", keys, err, row.want)
			}
		})
	}
}
