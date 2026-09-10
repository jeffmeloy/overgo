package gate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// runtimeInputs classifies how a package's sources reach beyond their own
// directory at run time, read from the literals in its execution and file
// calls: repository commands it runs, repository paths it names, and the
// reasons a dynamic reach keeps the broad binding.
type runtimeInputs struct {
	commands []string
	// files are repository paths the compiled sources name; testFiles are
	// paths only the package's tests name, inputs of those tests alone.
	files     []string
	testFiles []string
	// dynamic reasons come from compiled sources, so importers observe the
	// unnamed reach; testDynamic reasons come from tests alone, so only the
	// package's own tests must run on any change.
	dynamic     []string
	testDynamic []string
	// escapes records evidence that the sources reach the repository root:
	// a parent path, the working directory, the caller's file or a git
	// query. Without it a fixture root is a temporary directory.
	escapes bool
}

// confined reports a package whose runtime reach stays inside its own
// directory, external tools and temporary files.
func (inputs runtimeInputs) confined() bool {
	return len(inputs.commands) == 0 && len(inputs.files) == 0 && len(inputs.testFiles) == 0 && len(inputs.dynamic) == 0 && len(inputs.testDynamic) == 0
}

// execFunctions name the calls that start a process; the context variant
// takes the program after the context.
var execFunctions = []string{"exec.Command", "exec.CommandContext"}

// fileOwners are the packages whose calls take file system paths.
var fileOwners = []string{"os", "filepath", "ioutil", "fs"}

// temporaryCalls name the calls that yield a path outside the repository.
var temporaryCalls = []string{"TempDir", "MkdirTemp", "CreateTemp"}

// discoveryCall names a callee that reads a directory tree or a file by a
// path it is handed, outside the file owners: repository discovery, walks,
// globs, loads and reads.
var discoveryCall = regexp.MustCompile(`(?i)(discover|walk|glob|read|load|open|scan|stat|list)`)

// repositoryRoots are the top-level entries a literal path may name to
// reach repository inputs outside its package.
var repositoryRoots = []string{"cmd", "internal", "docs", "kernels", "web", "tmp", "build", "scripts", "testdata"}

// sourceClassifier carries one source file's classification state.
type sourceClassifier struct {
	inputs      *runtimeInputs
	relativeDir string
	file        string
	commands    map[string]bool
	named       map[string]bool
	pending     map[string]bool
	reasons     *[]string
	// safeNames are identifiers that hold the running binary's own path or
	// a temporary path, which reach nothing in the repository.
	safeNames map[string]bool
	// parameters are the enclosing function's receiver and parameters; a
	// path or program rooted in one is the caller's reach, which the
	// package may declare with the runtime-inputs directive.
	parameters  map[string]bool
	callerReach bool
	escapes     *bool
}

// runtimeInputsDirective declares, in any comment of a package, that the
// paths and programs its functions take from callers are the callers' reach:
// the package itself reads and runs nothing it does not name.
const runtimeInputsDirective = "//overgo:runtime-inputs caller"

// classifyRuntimeInputs parses the package's compiled and test sources under
// dir and classifies every execution call, every path handed to a file or
// discovery call, and every literal that names a repository file outright.
func classifyRuntimeInputs(root, dir string, files []string) (runtimeInputs, error) {
	var inputs runtimeInputs
	set := token.NewFileSet()
	commands := map[string]bool{}
	paths, testPaths := map[string]bool{}, map[string]bool{}
	// A bare repository root handed to a discovery call names the tree only
	// when the sources also reach the repository root; a fixture under a
	// temporary directory uses the same words for its own tree.
	trees, testTrees := map[string]bool{}, map[string]bool{}
	escapes := false
	relativeDir, err := filepath.Rel(root, dir)
	if err != nil {
		return runtimeInputs{}, err
	}
	relativeDir = filepath.ToSlash(relativeDir)
	parsedFiles := make([]*ast.File, 0, len(files))
	callerReach := false
	for _, name := range files {
		parsed, err := parser.ParseFile(set, filepath.Join(dir, name), nil, parser.ParseComments)
		if err != nil {
			return runtimeInputs{}, err
		}
		parsedFiles = append(parsedFiles, parsed)
		for _, group := range parsed.Comments {
			for _, comment := range group.List {
				callerReach = callerReach || strings.TrimSpace(comment.Text) == runtimeInputsDirective
			}
		}
	}
	// Package-level constants are literals by another name in every file.
	constants := map[string]bool{}
	for _, parsed := range parsedFiles {
		for _, declaration := range parsed.Decls {
			generic, ok := declaration.(*ast.GenDecl)
			if !ok || generic.Tok != token.CONST {
				continue
			}
			for _, spec := range generic.Specs {
				if value, ok := spec.(*ast.ValueSpec); ok {
					for _, name := range value.Names {
						constants[name.Name] = true
					}
				}
			}
		}
	}
	for index, name := range files {
		parsed := parsedFiles[index]
		classifier := sourceClassifier{
			inputs: &inputs, relativeDir: relativeDir, file: name, commands: commands,
			named: paths, pending: trees, reasons: &inputs.dynamic,
			safeNames: safeNames(parsed, constants), parameters: map[string]bool{}, callerReach: callerReach, escapes: &escapes,
		}
		if strings.HasSuffix(name, "_test.go") {
			classifier.named, classifier.pending, classifier.reasons = testPaths, testTrees, &inputs.testDynamic
		}
		ast.Inspect(parsed, classifier.visit)
	}
	inputs.escapes = escapes
	if escapes {
		maps.Copy(paths, trees)
		maps.Copy(testPaths, testTrees)
	}
	inputs.commands = slices.Sorted(maps.Keys(commands))
	inputs.files = slices.Sorted(maps.Keys(paths))
	for path := range testPaths {
		if !paths[path] {
			inputs.testFiles = append(inputs.testFiles, path)
		}
	}
	slices.Sort(inputs.testFiles)
	for _, reasons := range []*[]string{&inputs.dynamic, &inputs.testDynamic} {
		slices.Sort(*reasons)
		*reasons = slices.Compact(*reasons)
	}
	return inputs, nil
}

// visit classifies one syntax node.
func (classifier *sourceClassifier) visit(node ast.Node) bool {
	switch typed := node.(type) {
	case *ast.FuncDecl:
		classifier.parameters = functionParameters(typed.Recv, typed.Type)
	case *ast.FuncLit:
		maps.Copy(classifier.parameters, functionParameters(nil, typed.Type))
	case *ast.RangeStmt:
		// Elements of a caller-derived collection are caller-derived.
		if classifier.callerDerived(typed.X) {
			for _, target := range []ast.Expr{typed.Key, typed.Value} {
				if identifier, ok := target.(*ast.Ident); ok {
					classifier.parameters[identifier.Name] = true
				}
			}
		}
	case *ast.AssignStmt:
		// A local built from parameters or other caller-derived names is
		// the caller's reach too.
		derived := len(typed.Rhs) != 0
		for _, value := range typed.Rhs {
			derived = derived && classifier.callerDerived(value)
		}
		if derived {
			for _, target := range typed.Lhs {
				if identifier, ok := target.(*ast.Ident); ok {
					classifier.parameters[identifier.Name] = true
				}
			}
		}
	case *ast.CallExpr:
		if program, isExec := execProgram(typed); isExec {
			classifier.classifyProgram(program)
			for _, value := range literalArguments(typed) {
				*classifier.escapes = *classifier.escapes || value == "rev-parse"
			}
			return true
		}
		if selector, ok := typed.Fun.(*ast.SelectorExpr); ok && (selectorIs(selector, "runtime", "Caller") || selectorIs(selector, "os", "Getwd")) {
			*classifier.escapes = true
		}
		if pathCall(typed) {
			for _, value := range literalArguments(typed) {
				classifier.classifyPath(value, false)
			}
			// A file owner's call takes its path first (a join takes only
			// paths); a path it is handed that the source does not name may
			// reach any repository file.
			if fileOwnerCall(typed) && !classifier.safePath(typed.Args[0]) {
				if root := rootIdentifier(typed.Args[0]); root != "" && classifier.parameters[root] {
					if !classifier.callerReach {
						*classifier.reasons = append(*classifier.reasons, classifier.file+": reads a path a caller hands in")
					}
				} else {
					*classifier.reasons = append(*classifier.reasons, classifier.file+": reads a path the source does not name")
				}
			}
		}
	case *ast.BasicLit:
		if typed.Kind == token.STRING {
			if value, err := strconv.Unquote(typed.Value); err == nil {
				*classifier.escapes = *classifier.escapes || escapesPackage(value, classifier.relativeDir)
				classifier.classifyPath(value, true)
			}
		}
	}
	return true
}

// safePath reports a path argument that reaches nothing in the repository:
// a string literal, a temporary path, the binary's own path, a join of such
// parts, or a flag, mode or handle rather than a path.
func (classifier *sourceClassifier) safePath(argument ast.Expr) bool {
	switch typed := argument.(type) {
	case *ast.BasicLit:
		return true
	case *ast.Ident:
		return classifier.safeNames[typed.Name] || typed.Name == "nil"
	case *ast.CallExpr:
		if selector, ok := typed.Fun.(*ast.SelectorExpr); ok {
			if slices.Contains(temporaryCalls, selector.Sel.Name) {
				return true
			}
			if selectorIs(selector, "filepath", "Join") || selectorIs(selector, "path", "Join") {
				return !slices.ContainsFunc(typed.Args, func(part ast.Expr) bool { return !classifier.safePath(part) })
			}
		}
		return false
	case *ast.UnaryExpr, *ast.CompositeLit, *ast.FuncLit:
		// A flag, a mode, a callback or a literal value, never a path.
		return true
	default:
		root := rootIdentifier(argument)
		return root != "" && classifier.safeNames[root]
	}
}

// safeNames collects the identifiers a file assigns from the running
// binary's own path or from a temporary directory or file.
func safeNames(file *ast.File, constants map[string]bool) map[string]bool {
	names := maps.Clone(constants)
	ast.Inspect(file, func(node ast.Node) bool {
		assign, ok := node.(*ast.AssignStmt)
		if !ok || len(assign.Rhs) == 0 {
			return true
		}
		safe := false
		for _, value := range assign.Rhs {
			safe = safe || safeSource(value, names)
		}
		if !safe {
			return true
		}
		for _, target := range assign.Lhs {
			if identifier, ok := target.(*ast.Ident); ok {
				names[identifier.Name] = true
			}
		}
		return true
	})
	return names
}

// safeSource reports a value that holds the binary's own path, a temporary
// path, or a join of literals and such names.
func safeSource(value ast.Expr, names map[string]bool) bool {
	switch typed := value.(type) {
	case *ast.BasicLit:
		return typed.Kind == token.STRING
	case *ast.Ident:
		return names[typed.Name]
	case *ast.IndexExpr:
		selector, ok := typed.X.(*ast.SelectorExpr)
		return ok && selectorIs(selector, "os", "Args")
	case *ast.CallExpr:
		selector, ok := typed.Fun.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		if selectorIs(selector, "os", "Executable") || slices.Contains(temporaryCalls, selector.Sel.Name) {
			return true
		}
		if selectorIs(selector, "filepath", "Join") || selectorIs(selector, "path", "Join") {
			return !slices.ContainsFunc(typed.Args, func(part ast.Expr) bool { return !safeSource(part, names) })
		}
	}
	return false
}

// execProgram returns the program expression of an exec call.
func execProgram(call *ast.CallExpr) (ast.Expr, bool) {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil, false
	}
	owner, ok := selector.X.(*ast.Ident)
	if !ok || !slices.Contains(execFunctions, owner.Name+"."+selector.Sel.Name) || len(call.Args) == 0 {
		return nil, false
	}
	if !strings.HasSuffix(selector.Sel.Name, "Context") {
		return call.Args[0], true
	}
	if len(call.Args) <= 1 {
		return nil, false
	}
	return call.Args[1], true
}

// fileOwnerCall reports a call on a file owner with at least one argument.
func fileOwnerCall(call *ast.CallExpr) bool {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || len(call.Args) == 0 {
		return false
	}
	owner, ok := selector.X.(*ast.Ident)
	return ok && slices.Contains(fileOwners, owner.Name)
}

// pathCall reports a call whose arguments are file system paths: a file
// owner's function or a discovery-shaped callee.
func pathCall(call *ast.CallExpr) bool {
	switch function := call.Fun.(type) {
	case *ast.SelectorExpr:
		if owner, ok := function.X.(*ast.Ident); ok && slices.Contains(fileOwners, owner.Name) {
			return true
		}
		return discoveryCall.MatchString(function.Sel.Name)
	case *ast.Ident:
		return discoveryCall.MatchString(function.Name)
	}
	return false
}

// literalArguments lists the string literals among a call's arguments,
// looking through nested path joins.
func literalArguments(call *ast.CallExpr) []string {
	var values []string
	for _, argument := range call.Args {
		switch typed := argument.(type) {
		case *ast.BasicLit:
			if typed.Kind == token.STRING {
				if value, err := strconv.Unquote(typed.Value); err == nil {
					values = append(values, value)
				}
			}
		case *ast.CallExpr:
			if selector, ok := typed.Fun.(*ast.SelectorExpr); ok && selectorIs(selector, "filepath", "Join") || ok && selectorIs(selector, "path", "Join") {
				values = append(values, literalArguments(typed)...)
			}
		}
	}
	return values
}

// classifyProgram decides what an executed program reaches: the test binary
// itself and named external tools reach nothing in the repository, a literal
// repository command binds its package, and anything else is dynamic.
func (classifier *sourceClassifier) classifyProgram(program ast.Expr) {
	switch typed := program.(type) {
	case *ast.IndexExpr:
		if selector, ok := typed.X.(*ast.SelectorExpr); ok && selectorIs(selector, "os", "Args") {
			return
		}
	case *ast.BasicLit:
		if typed.Kind == token.STRING {
			value, err := strconv.Unquote(typed.Value)
			if err == nil && !strings.ContainsAny(value, `/\`) {
				return
			}
			if err == nil {
				if command := repositoryCommand(value, classifier.relativeDir); command != "" {
					classifier.commands[command] = true
					return
				}
			}
		}
	}
	root := rootIdentifier(program)
	if root != "" && classifier.safeNames[root] {
		return
	}
	if root != "" && classifier.parameters[root] {
		if !classifier.callerReach {
			*classifier.reasons = append(*classifier.reasons, classifier.file+": executes a program a caller hands in")
		}
		return
	}
	*classifier.reasons = append(*classifier.reasons, classifier.file+": executes a program the source does not name")
}

// callerDerived reports a value built only from literals, parameters,
// caller-derived names, safe names and calls or selections on them.
func (classifier *sourceClassifier) callerDerived(value ast.Expr) bool {
	switch typed := value.(type) {
	case *ast.BasicLit:
		return true
	case *ast.Ident:
		return classifier.parameters[typed.Name] || classifier.safeNames[typed.Name] || typed.Name == "nil"
	case *ast.CallExpr:
		derivedArguments := !slices.ContainsFunc(typed.Args, func(part ast.Expr) bool { return !classifier.callerDerived(part) })
		if selector, ok := typed.Fun.(*ast.SelectorExpr); ok {
			if slices.Contains(temporaryCalls, selector.Sel.Name) {
				return true
			}
			if owner, ok := selector.X.(*ast.Ident); ok && slices.Contains(derivingOwners, owner.Name) {
				// The environment and the working directory come from
				// outside the call, never from the caller.
				return derivedArguments && !slices.Contains(environmentCalls, selector.Sel.Name)
			}
		}
		// A function over caller-derived arguments yields a caller-derived
		// value; a function that takes nothing may find the repository on
		// its own.
		if len(typed.Args) != 0 && derivedArguments {
			return true
		}
		root := rootIdentifier(typed.Fun)
		return root != "" && (classifier.parameters[root] || classifier.safeNames[root]) && derivedArguments
	case *ast.SelectorExpr, *ast.IndexExpr, *ast.StarExpr, *ast.ParenExpr:
		root := rootIdentifier(value)
		return root != "" && (classifier.parameters[root] || classifier.safeNames[root])
	case *ast.UnaryExpr:
		return classifier.callerDerived(typed.X)
	case *ast.BinaryExpr:
		return classifier.callerDerived(typed.X) && classifier.callerDerived(typed.Y)
	case *ast.CompositeLit:
		return !slices.ContainsFunc(typed.Elts, func(element ast.Expr) bool {
			if pair, ok := element.(*ast.KeyValueExpr); ok {
				element = pair.Value
			}
			return !classifier.callerDerived(element)
		})
	default:
		return false
	}
}

// derivingOwners are the standard packages whose functions return values
// derived only from their arguments.
var derivingOwners = []string{"os", "filepath", "path", "strings", "fmt", "io", "bufio", "bytes", "strconv", "slices", "errors", "sort"}

// environmentCalls name the os functions whose results come from the
// process environment rather than from the caller.
var environmentCalls = []string{"Getenv", "LookupEnv", "Environ", "Getwd", "UserHomeDir", "UserCacheDir", "UserConfigDir"}

// functionParameters names a function's receiver and parameters.
func functionParameters(receiver *ast.FieldList, signature *ast.FuncType) map[string]bool {
	names := map[string]bool{}
	for _, list := range []*ast.FieldList{receiver, signature.Params} {
		if list == nil {
			continue
		}
		for _, field := range list.List {
			for _, name := range field.Names {
				names[name.Name] = true
			}
		}
	}
	return names
}

// rootIdentifier returns the identifier an expression is rooted in through
// selectors, indexes and calls on it; empty when there is none.
func rootIdentifier(expression ast.Expr) string {
	for {
		switch typed := expression.(type) {
		case *ast.Ident:
			return typed.Name
		case *ast.SelectorExpr:
			expression = typed.X
		case *ast.IndexExpr:
			expression = typed.X
		case *ast.CallExpr:
			expression = typed.Fun
		case *ast.ParenExpr:
			expression = typed.X
		case *ast.StarExpr:
			expression = typed.X
		default:
			return ""
		}
	}
}

// classifyPath binds a literal that names a repository path outside the
// package. A command package is unambiguous wherever it is named, since a
// process wrapper runs it on this package's behalf. Handed to a file or
// discovery call, a directory binds the tree it names, a bare repository
// root only pending evidence that the sources reach the repository root;
// anywhere else only a literal that names a file outright, a slash path
// with an extension, binds. A package pattern is dynamic.
func (classifier *sourceClassifier) classifyPath(value string, literalOnly bool) {
	first, _, _ := strings.Cut(strings.TrimSpace(value), " ")
	cleaned := path.Clean(strings.ReplaceAll(first, `\`, "/"))
	if strings.HasSuffix(cleaned, "/...") || strings.HasSuffix(strings.TrimSpace(value), "/...") {
		*classifier.reasons = append(*classifier.reasons, "names the package pattern "+value)
		return
	}
	if command := repositoryCommand(cleaned, classifier.relativeDir); command != "" {
		classifier.commands[command] = true
		return
	}
	if literalOnly && (!strings.Contains(cleaned, "/") || path.Ext(cleaned) == "") {
		return
	}
	resolved, ok := repositoryPath(cleaned, classifier.relativeDir)
	if !ok {
		return
	}
	if !strings.Contains(resolved, "/") && !strings.HasPrefix(cleaned, "../") {
		classifier.pending[resolved] = true
		return
	}
	classifier.named[resolved] = true
}

// escapesPackage reports a literal that walks out of the package directory
// toward the repository root.
func escapesPackage(value, relativeDir string) bool {
	cleaned := path.Clean(strings.ReplaceAll(value, `\`, "/"))
	if cleaned == ".." {
		return true
	}
	if !strings.HasPrefix(cleaned, "../") {
		return false
	}
	_, ok := repositoryPath(cleaned, relativeDir)
	return ok
}

// repositoryCommand resolves a literal naming a command package under cmd
// to its repo-relative directory, following ../ from the package directory.
func repositoryCommand(value, relativeDir string) string {
	resolved, ok := repositoryPath(path.Clean(strings.ReplaceAll(value, `\`, "/")), relativeDir)
	if !ok || !strings.HasPrefix(resolved, "cmd/") || strings.Count(resolved, "/") != 1 || strings.ContainsAny(resolved, " \t") {
		return ""
	}
	return resolved
}

// repositoryPath resolves a literal to a repo-relative path when it names a
// repository entry outside the package directory: an explicit ./ or ../
// prefix resolves from the package, a bare top-level root resolves from the
// repository root. Paths inside the package and everything else resolve to
// nothing.
func repositoryPath(cleaned, relativeDir string) (string, bool) {
	if cleaned == "" || cleaned == "." || path.IsAbs(cleaned) || strings.Contains(cleaned, ":") {
		return "", false
	}
	var resolved string
	switch {
	case strings.HasPrefix(cleaned, "../"):
		resolved = path.Clean(path.Join(relativeDir, cleaned))
		if resolved == "." || strings.HasPrefix(resolved, "../") {
			return "", false
		}
	case strings.HasPrefix(cleaned, "./"):
		return "", false
	default:
		first, _, _ := strings.Cut(cleaned, "/")
		if !slices.Contains(repositoryRoots, first) || first == "testdata" {
			return "", false
		}
		resolved = cleaned
	}
	if resolved == relativeDir || strings.HasPrefix(resolved, relativeDir+"/") {
		return "", false
	}
	return resolved, true
}

// namesPath reports whether a repo-relative path is one of the named
// repository paths or lies under a named directory.
func namesPath(named []string, name string) bool {
	for _, entry := range named {
		if name == entry {
			return true
		}
		if strings.HasPrefix(name, entry) && len(name) > len(entry) && name[len(entry)] == '/' {
			return true
		}
	}
	return false
}

// selectorIs reports whether a selector names owner.member.
func selectorIs(selector *ast.SelectorExpr, owner, member string) bool {
	if selector.Sel.Name != member {
		return false
	}
	identifier, ok := selector.X.(*ast.Ident)
	return ok && identifier.Name == owner
}

// packageSources lists the compiled and test Go files go list reported for
// a package that exist on disk.
func packageSources(node goPackageInput) []string {
	var files []string
	for _, name := range slices.Concat(node.GoFiles, node.CgoFiles, node.TestGoFiles, node.XTestGoFiles) {
		if _, err := os.Stat(filepath.Join(node.Dir, name)); err == nil {
			files = append(files, name)
		}
	}
	return files
}
