package gate

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

type goModuleInput struct {
	GoMod string
}

type goPackageInput struct {
	ImportPath      string
	ForTest         string
	Match           []string
	Dir             string
	Standard        bool
	Module          *goModuleInput
	Imports         []string
	TestImports     []string
	XTestImports    []string
	GoFiles         []string
	CgoFiles        []string
	CFiles          []string
	CXXFiles        []string
	MFiles          []string
	HFiles          []string
	FFiles          []string
	SFiles          []string
	SwigFiles       []string
	SwigCXXFiles    []string
	SysoFiles       []string
	EmbedFiles      []string
	TestGoFiles     []string
	XTestGoFiles    []string
	TestEmbedFiles  []string
	XTestEmbedFiles []string
	// Non-import edges derived from runtime inputs: the command packages
	// the source names, or every root when the reach is not named.
	inputDependencies []string
	// Test commands and opaque reads belong to the package's own test execution.
	testInputDependencies []string
	// Execution edges alone propagate device requirements.
	executionDependencies []string
	// opaqueReader marks a package whose compiled runtime reach the source
	// does not name, so it and its importers may observe any candidate
	// source; testOpaque marks the same for the package's tests alone, so
	// only its own tests run on any change. The reason is audited.
	opaqueReader bool
	testOpaque   bool
	// sourceReader marks a package whose compiled sources observe Go source
	// as data: they parse or format Go, or run a program the source does
	// not name, which may be any repository command. Any other unnamed
	// reach binds the repository's data files, never its sources.
	// testSourceReader marks the same for the package's tests alone.
	sourceReader     bool
	testSourceReader bool
	// runtimeReason explains a broad binding: what the source left unnamed.
	runtimeReason string
	// Named repository inputs retain runtime acceptance even when Markdown.
	declaredFiles []string
	// productionFiles are the named inputs of the compiled sources alone;
	// an importer inherits these, never the inputs the tests name.
	namedProductionFiles []string
}

type packageInputGraph struct {
	root          string
	nodes         []goPackageInput
	byID          map[string][]int
	resourceFiles map[string][]string
	// Includes tracked Go files omitted by the host build selection.
	sourceDirectories map[string]bool
	// testResourceFiles are repository paths a package's tests name at run
	// time: inputs of that package's tests alone, never of its importers.
	testResourceFiles map[string][]string
	// runtimeResourceFiles are repository paths a package's compiled
	// sources name at run time: inputs of the package's own tests, and of
	// importers whose tests reach the repository root.
	runtimeResourceFiles map[string][]string
	// Scoped to one identity batch; never retained across candidate reads.
	fileInputs map[string][]byte
}

// Presence tags frame the v3 cache input grammar independently of file bytes.
const (
	packageInputAbsent byte = iota
	packageInputPresent
)

func loadPackageInputGraph(root string) (packageInputGraph, error) {
	out, err := command(root, "go", "list", "-deps", "-test", "-json", "./...")
	if err != nil {
		return packageInputGraph{}, err
	}
	decoder := json.NewDecoder(bytes.NewBufferString(out))
	graph := packageInputGraph{root: root, byID: map[string][]int{}}
	for {
		var node goPackageInput
		if err := decoder.Decode(&node); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return packageInputGraph{}, fmt.Errorf("decode package inputs: %w", err)
		}
		index := len(graph.nodes)
		graph.nodes = append(graph.nodes, node)
		graph.byID[node.ImportPath] = append(graph.byID[node.ImportPath], index)
	}
	paths, err := gitLines(root, "ls-files", "-co", "--exclude-standard")
	if err != nil {
		return packageInputGraph{}, fmt.Errorf("derive repository inputs: %w", err)
	}
	graph.bindResourceFiles(paths)
	if gatePackageGraphLoadedHook != nil {
		gatePackageGraphLoadedHook(root, len(graph.nodes))
	}
	return graph, nil
}

// bindResourceFiles attaches undeclared repository assets to their nearest
// package. Compiler-declared inputs retain their production/test classification.
// This same ownership feeds selection and cache identity; ignored external data
// is outside the immutable source candidate and is not treated as source evidence.
func (graph *packageInputGraph) bindResourceFiles(paths []string) {
	if graph.resourceFiles == nil {
		graph.resourceFiles = map[string][]string{}
	}
	if graph.sourceDirectories == nil {
		graph.sourceDirectories = map[string]bool{}
	}
	for _, name := range paths {
		if strings.EqualFold(filepath.Ext(name), ".go") {
			graph.sourceDirectories[filepath.Dir(filepath.Join(graph.root, filepath.FromSlash(name)))] = true
		}
	}
	compiled := map[string]bool{}
	var directories, roots []string
	for _, node := range graph.nodes {
		if len(node.Match) == 0 || node.ForTest != "" {
			continue
		}
		directories = append(directories, node.Dir)
		roots = append(roots, node.ImportPath)
		for _, name := range node.files() {
			compiled[filepath.Join(node.Dir, filepath.FromSlash(name))] = true
		}
	}
	// Commands and file readers reach beyond their package at run time. A
	// package whose source names what it runs and reads binds only those
	// edges: the named command packages and the named repository paths. A
	// program or path the source does not name keeps the conservative
	// binding to every root until a declared runtime contract narrows it.
	byDirectory := map[string]string{}
	for _, node := range graph.nodes {
		if len(node.Match) == 0 || node.ForTest != "" {
			continue
		}
		if relative, err := filepath.Rel(graph.root, node.Dir); err == nil {
			byDirectory[filepath.ToSlash(relative)] = node.ImportPath
		}
	}
	var runtimeDirectories []string
	broadTest := map[string]bool{}
	namedPaths, namedTestPaths := map[string][]string{}, map[string][]string{}
	if graph.testResourceFiles == nil {
		graph.testResourceFiles = map[string][]string{}
	}
	if graph.runtimeResourceFiles == nil {
		graph.runtimeResourceFiles = map[string][]string{}
	}
	for index := range graph.nodes {
		node := &graph.nodes[index]
		if !slices.Contains(directories, node.Dir) || len(node.Match) == 0 && node.ForTest == "" {
			continue
		}
		inputs, err := classifyRuntimeInputs(graph.root, node.Dir, packageSources(*node))
		node.declaredFiles = slices.Concat(inputs.files, inputs.testFiles)
		node.namedProductionFiles = slices.Clone(inputs.files)
		// Tests read and run at run time as production code does; a package
		// whose tests alone import os is a runtime reader of its own tests.
		edges := slices.Concat(node.Imports, node.TestImports, node.XTestImports)
		runtimeReader := slices.Contains(edges, "os/exec") || slices.Contains(edges, "os") || slices.Contains(edges, "io/ioutil")
		if !runtimeReader {
			continue
		}
		node.sourceReader = inputs.dynamicExec || slices.ContainsFunc(node.Imports, sourceReaderImport)
		node.testSourceReader = node.sourceReader || inputs.testDynamicExec ||
			slices.ContainsFunc(slices.Concat(node.TestImports, node.XTestImports), sourceReaderImport)
		testSourceReader := node.testSourceReader
		reason := ""
		switch {
		case err != nil:
			reason = "runtime inputs unreadable: " + err.Error()
		case len(inputs.dynamic) != 0:
			reason = strings.Join(inputs.dynamic, "; ")
		}
		absent := ""
		if reason == "" {
			// A named command the graph does not hold (deleted or renamed)
			// leaves the consumer's reach unexplained: its tests run on any
			// change until the name resolves again.
			for _, command := range inputs.commands {
				if target, ok := byDirectory[command]; ok {
					node.inputDependencies = append(node.inputDependencies, target)
					node.executionDependencies = append(node.executionDependencies, target)
				} else {
					absent = "runs the command " + command + " the graph does not hold"
				}
			}
			for _, command := range inputs.testCommands {
				if target, ok := byDirectory[command]; ok {
					node.testInputDependencies = append(node.testInputDependencies, target)
					node.executionDependencies = append(node.executionDependencies, target)
				} else {
					inputs.testDynamic = append(inputs.testDynamic, "runs the command "+command+" the graph does not hold")
				}
			}
		}
		if reason == "" && absent != "" {
			reason = absent
		}
		if reason == "" && absent == "" && inputs.confined() {
			// Temporary files, the test binary and external tools: no edge.
			continue
		}
		// A named path binds to its namer whether or not the package also
		// reads what it does not name; a named document then leaves the
		// broad binding below.
		namedPaths[node.Dir] = append(namedPaths[node.Dir], inputs.files...)
		namedTestPaths[node.Dir] = append(namedTestPaths[node.Dir], inputs.testFiles...)
		if reason == "" {
			if len(inputs.testDynamic) != 0 {
				// The tests alone reach what they do not name: they run on
				// any change, but importers observe only the compiled reach.
				node.testOpaque = true
				node.runtimeReason = "tests: " + strings.Join(inputs.testDynamic, "; ")
				if testSourceReader {
					node.testInputDependencies = slices.Clone(roots)
				}
				if !slices.Contains(runtimeDirectories, node.Dir) {
					runtimeDirectories = append(runtimeDirectories, node.Dir)
					broadTest[node.Dir] = true
				}
			}
			continue
		}
		node.opaqueReader = true
		node.runtimeReason = reason
		// Only a Go-parsing reach observes every root's sources; any other
		// unnamed reach binds the data files the broad resource loop adds.
		if node.sourceReader {
			node.inputDependencies = slices.Clone(roots)
		}
		if slices.Contains(edges, "os/exec") {
			node.executionDependencies = node.inputDependencies
		}
		if !slices.Contains(runtimeDirectories, node.Dir) {
			runtimeDirectories = append(runtimeDirectories, node.Dir)
		}
	}
	for _, name := range paths {
		absolute := filepath.Join(graph.root, filepath.FromSlash(name))
		if documentationChanges([]string{name}, *graph) {
			continue
		}
		// A reader that names a source tree consumes its Go files as data;
		// compiled inputs otherwise arrive through imports alone.
		var owners []string
		for directory, named := range namedPaths {
			if namesPath(named, name) && !slices.Contains(graph.runtimeResourceFiles[directory], absolute) {
				graph.runtimeResourceFiles[directory] = append(graph.runtimeResourceFiles[directory], absolute)
			}
		}
		for directory, named := range namedTestPaths {
			if namesPath(named, name) && !slices.Contains(graph.testResourceFiles[directory], absolute) {
				graph.testResourceFiles[directory] = append(graph.testResourceFiles[directory], absolute)
			}
		}
		if !strings.EqualFold(filepath.Ext(name), ".go") {
			owner := ""
			for _, directory := range directories {
				relative, err := filepath.Rel(directory, absolute)
				if err == nil && relative != ".." && !filepath.IsAbs(relative) && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && len(directory) > len(owner) {
					owner = directory
				}
			}
			// A reader whose reach the source does not name may read any
			// repository file: its tests' input when only the tests are
			// unnamed, otherwise a runtime input that taints importers whose
			// tests reach the repository. Adjacent assets stay compiled inputs.
			// A document some package names is bound above to its namers
			// alone; an unnamed read pointed at it is outside the boundary.
			broad := runtimeDirectories
			if namedDocument(namedPaths, namedTestPaths, name) {
				broad = nil
			}
			for _, directory := range broad {
				target := graph.runtimeResourceFiles
				if broadTest[directory] {
					target = graph.testResourceFiles
				}
				if !slices.Contains(target[directory], absolute) {
					target[directory] = append(target[directory], absolute)
				}
			}
			if owner != "" && !compiled[absolute] && !slices.Contains(owners, owner) {
				owners = append(owners, owner)
			}
		}
		for _, directory := range owners {
			if !slices.Contains(graph.resourceFiles[directory], absolute) {
				graph.resourceFiles[directory] = append(graph.resourceFiles[directory], absolute)
			}
		}
	}
}

// dependentDirectories returns every repository package that is or transitively
// imports a package rooted in one of the declared repository directories.
func (graph packageInputGraph) dependentDirectories(roots ...string) ([]string, error) {
	owned := map[string]bool{}
	for _, node := range graph.nodes {
		relative, err := filepath.Rel(graph.root, node.Dir)
		if err != nil {
			return nil, err
		}
		relative = filepath.ToSlash(relative)
		for _, root := range roots {
			if relative == root || strings.HasPrefix(relative, root+"/") {
				owned[node.ImportPath] = true
				break
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, node := range graph.nodes {
			if owned[node.ImportPath] {
				continue
			}
			for _, imported := range slices.Concat(node.Imports, node.TestImports, node.XTestImports, node.executionDependencies) {
				if owned[imported] {
					owned[node.ImportPath] = true
					changed = true
					break
				}
			}
		}
	}
	directories := map[string]bool{}
	for _, node := range graph.nodes {
		if !owned[node.ImportPath] {
			continue
		}
		relative, err := filepath.Rel(graph.root, node.Dir)
		if err != nil {
			return nil, err
		}
		if relative == ".." || filepath.IsAbs(relative) || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			continue
		}
		directories[filepath.ToSlash(relative)] = true
	}
	result := make([]string, 0, len(directories))
	for directory := range directories {
		result = append(result, directory)
	}
	slices.Sort(result)
	return result, nil
}

// inputNodes follows the identity owner's edges. Compiler-only traversal is a
// diagnostic view; it cannot authorize test exclusion or evidence reuse.
func (graph packageInputGraph) inputNodes(target string, runtimeInputs bool) (map[int]bool, error) {
	return graph.closureNodes(target, runtimeInputs, true)
}

// closureNodes walks the target's imports, its own test imports, and the
// runtime edges of every node reached. With testRuntime false the runtime
// edges of a node reached only through the target's test imports are not
// followed: a Go-parsing or program-running helper that only the tests
// import is the uncertain reach the scope runs short, not a lane's reach.
func (graph packageInputGraph) closureNodes(target string, runtimeInputs, testRuntime bool) (map[int]bool, error) {
	queue := append([]int(nil), graph.byID[target]...)
	for index, node := range graph.nodes {
		if node.ForTest == target {
			queue = append(queue, index)
		}
	}
	if len(queue) == 0 {
		return nil, fmt.Errorf("package input identity: package %q is absent", target)
	}
	withTests := map[int]bool{}
	compiled := map[int]bool{}
	var visit func(int, bool, bool)
	visit = func(index int, tests, viaTest bool) {
		prior, seen := withTests[index]
		if seen && (!tests || prior) && (viaTest || compiled[index]) {
			return
		}
		withTests[index] = prior || tests
		if !viaTest {
			compiled[index] = true
		}
		for _, imported := range graph.nodes[index].Imports {
			for _, dependency := range graph.byID[imported] {
				visit(dependency, false, viaTest)
			}
		}
		if tests {
			for _, imported := range slices.Concat(graph.nodes[index].TestImports, graph.nodes[index].XTestImports) {
				for _, dependency := range graph.byID[imported] {
					visit(dependency, false, true)
				}
			}
		}
		if !runtimeInputs {
			return
		}
		// A named command a test helper runs is a real reach; the unnamed
		// Go-source binding of a test-only parser or runner is not.
		var runtime []string
		if !(viaTest && !testRuntime && graph.nodes[index].sourceReader) {
			runtime = slices.Clone(graph.nodes[index].inputDependencies)
		}
		if tests && !(viaTest && !testRuntime && graph.nodes[index].testSourceReader) {
			runtime = append(runtime, graph.nodes[index].testInputDependencies...)
		}
		for _, imported := range runtime {
			for _, dependency := range graph.byID[imported] {
				// Runtime commands may run tests; opaque readers may read test source.
				visit(dependency, true, viaTest)
			}
		}
	}
	for _, index := range queue {
		// A compiler-generated test variant imports the test's imports as
		// its own; they are test imports for the reach.
		visit(index, true, graph.nodes[index].ForTest != "")
	}
	return withTests, nil
}

// laneInputFiles lists the files a lane's owned package reaches: what it
// compiles, what its tests compile, and the runtime reach of its compiled
// closure. A test-only helper's unnamed reach is not among them.
func (graph packageInputGraph) laneInputFiles(target string) (map[string]bool, error) {
	withTests, err := graph.closureNodes(target, true, false)
	if err != nil {
		return nil, err
	}
	return graph.closureFiles(withTests)
}

func (graph packageInputGraph) inputFiles(target string) (map[string]bool, error) {
	withTests, err := graph.inputNodes(target, true)
	if err != nil {
		return nil, err
	}
	return graph.closureFiles(withTests)
}

// closureFiles lists the files of a walked closure: each node's compiled
// sources (its tests too where the walk reached them with tests), the
// resources bound to its directory, and the module files.
func (graph packageInputGraph) closureFiles(withTests map[int]bool) (map[string]bool, error) {
	files := map[string]bool{}
	for index := range withTests {
		node := graph.nodes[index]
		names := node.productionFiles()
		if withTests[index] {
			names = node.files()
		}
		for _, name := range names {
			files[filepath.Join(node.Dir, filepath.FromSlash(name))] = true
		}
		for _, path := range graph.resourceFiles[node.Dir] {
			files[path] = true
		}
		// The target's own tests read what they name; an imported package's
		// test inputs do not reach this identity, and its runtime-named
		// inputs always do: the target's tests run the import's reads, and
		// nothing establishes their isolation from them.
		if withTests[index] {
			for _, path := range graph.testResourceFiles[node.Dir] {
				files[path] = true
			}
		}
		for _, path := range graph.runtimeResourceFiles[node.Dir] {
			files[path] = true
		}
		if node.Module != nil && node.Module.GoMod != "" {
			files[node.Module.GoMod] = true
		}
	}
	for _, name := range []string{"go.mod", "go.sum"} {
		path := filepath.Join(graph.root, name)
		if _, err := os.Stat(path); err == nil {
			files[path] = true
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
	}
	return files, nil
}

func (graph packageInputGraph) identity(target string) (artifact.ID, error) {
	files, err := graph.inputFiles(target)
	if err != nil {
		return artifact.ID{}, err
	}
	// Repository paths are logical inputs; temporary checkout locations are not.
	logical := make(map[string]string, len(files))
	var paths []string
	for path := range files {
		path = filepath.Clean(path)
		name := "external:" + filepath.ToSlash(path)
		if relative, err := filepath.Rel(graph.root, path); err == nil && filepath.IsLocal(relative) {
			name = "repository:" + filepath.ToSlash(relative)
		}
		logical[name] = path
		paths = append(paths, name)
	}
	slices.Sort(paths)
	hasher := sha256.New()
	hasher.Write([]byte("go-test-inputs/v3\x00"))
	hasher.Write([]byte(target))
	hasher.Write([]byte("\x00"))
	for _, name := range paths {
		path := logical[name]
		hasher.Write([]byte(name))
		hasher.Write([]byte("\x00"))
		input, found := graph.fileInputs[path]
		if !found {
			content, err := os.ReadFile(path)
			switch {
			case errors.Is(err, os.ErrNotExist):
				input = []byte{packageInputAbsent}
			case err != nil:
				return artifact.ID{}, fmt.Errorf("package input identity %s: %w", target, err)
			default:
				digest := sha256.Sum256(content)
				input = append([]byte{packageInputPresent}, digest[:]...)
			}
			if graph.fileInputs != nil {
				graph.fileInputs[path] = input
			}
		}
		hasher.Write(input) // Presence plus fixed-width digest preserves framing.
	}
	return artifact.IdentifyBytes(artifact.KindEvidence, hasher.Sum(nil))
}

func (node goPackageInput) productionFiles() []string {
	var files []string
	for _, group := range [][]string{
		node.GoFiles, node.CgoFiles, node.CFiles, node.CXXFiles, node.MFiles,
		node.HFiles, node.FFiles, node.SFiles, node.SwigFiles, node.SwigCXXFiles,
		node.SysoFiles, node.EmbedFiles,
	} {
		files = append(files, group...)
	}
	return files
}

func (node goPackageInput) files() []string {
	files := node.productionFiles()
	for _, group := range [][]string{node.TestGoFiles, node.XTestGoFiles, node.TestEmbedFiles, node.XTestEmbedFiles} {
		files = append(files, group...)
	}
	return files
}

func packageInputIdentities(graph packageInputGraph, packages []string) (map[string]artifact.ID, error) {
	graph.fileInputs = map[string][]byte{}
	identities := make(map[string]artifact.ID, len(packages))
	for _, packagePath := range packages {
		identity, err := graph.identity(packagePath)
		if err != nil {
			return nil, err
		}
		identities[packagePath] = identity
	}
	return identities, nil
}

// deviceRuntimeRoot holds the device runtime; deviceWorkerPackage creates
// every device context; deviceTestPackages are what a test imports to reach
// the device: the driver, the worker, or the kernel test gate that skips it
// outside the device lane.
const (
	deviceRuntimeRoot   = "overgo/internal/cuda"
	deviceWorkerPackage = deviceRuntimeRoot + "/device"
)

var deviceTestPackages = []string{deviceRuntimeRoot + "/driver", deviceWorkerPackage, deviceRuntimeRoot + "/testutil"}

// devicePackages names, among the packages given and in their order, the
// ones whose tests execute on the device: a runtime package, a package whose
// tests import the driver, the worker or the kernel test gate, or one whose
// tests run a command that creates device contexts. Linking the runtime is
// no need of its own: a test reaches a kernel only through the gate it
// declares. It is the one fact the batch admission and the batch order read.
func (graph packageInputGraph) devicePackages(packages []string) ([]string, error) {
	workers := graph.productionDependents(deviceWorkerPackage)
	needing := map[string]bool{}
	for _, node := range graph.nodes {
		if node.ForTest != "" || strings.HasSuffix(node.ImportPath, ".test") {
			continue
		}
		switch {
		case node.ImportPath == deviceRuntimeRoot || strings.HasPrefix(node.ImportPath, deviceRuntimeRoot+"/"):
		case slices.ContainsFunc(slices.Concat(node.TestImports, node.XTestImports), func(imported string) bool {
			return slices.Contains(deviceTestPackages, imported)
		}):
		case slices.ContainsFunc(node.executionDependencies, func(command string) bool { return workers[command] }):
		default:
			continue
		}
		needing[node.ImportPath] = true
	}
	var devices []string
	for _, pkg := range packages {
		if needing[pkg] {
			devices = append(devices, pkg)
		}
	}
	return devices, nil
}

// productionDependents reports the packages that are or through compiled
// imports alone depend on the package named.
func (graph packageInputGraph) productionDependents(root string) map[string]bool {
	owned := map[string]bool{root: true}
	for changed := true; changed; {
		changed = false
		for _, node := range graph.nodes {
			if owned[node.ImportPath] || node.ForTest != "" {
				continue
			}
			if slices.ContainsFunc(node.Imports, func(imported string) bool { return owned[imported] }) {
				owned[node.ImportPath] = true
				changed = true
			}
		}
	}
	return owned
}

// namedDocument reports a path some package names in production or test source.
func namedDocument(namedPaths, namedTestPaths map[string][]string, name string) bool {
	for _, named := range namedPaths {
		if namesPath(named, name) {
			return true
		}
	}
	for _, named := range namedTestPaths {
		if namesPath(named, name) {
			return true
		}
	}
	return false
}

// sourceReaderImport reports an import that parses or formats Go source.
func sourceReaderImport(edge string) bool {
	return strings.HasPrefix(edge, "go/") || strings.Contains(edge, "/x/tools/go/")
}
