package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/types"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// consolidatedGroup is one group of repeated logic that a common component replaced.
//   - functions = the file and the function of each place the group held, written as shapePlace writes them without its lines
//   - a file alone in functions = every function the file declares, for a group whose places a common component merged into one function
//   - cases = the case expressions, as the source writes them, of the clauses that held the places of a group inside one function's switch
//   - calls = the names the repeated logic called and the common component now calls in one place
type consolidatedGroup struct {
	name      string
	functions []string
	cases     []string
	calls     []string
}

// consolidatedGroups are the groups the plan's step shape-reading/real-repeats consolidated.
//   - device upload and feed binding -> device.AllocationSet and vqaserve.BindFeed
//   - device-scope allocation chains -> cudaScope.fill
//   - device-scope kernel loads -> cudaScope.functions
var consolidatedGroups = []consolidatedGroup{
	{
		name: "device upload and feed binding",
		functions: []string{
			"cmd/validate/parity_vqa_device.go vqaDevice",
			"cmd/validate/parity_vqa_device_decode.go runDeviceDecode",
			"cmd/validate/parity_vqa_device_merger.go runDeviceMerger",
			"cmd/validate/parity_vqa_device_prefill.go runDevicePrefill",
			"cmd/validate/parity_vqa_device_vision.go runDeviceVision",
			"internal/vqaserve/pipeline.go RunPipeline",
			"internal/vqaserve/pipeline.go runMerger",
			"internal/vqaserve/pipeline.go runPrefill",
			"internal/vqaserve/pipeline.go runVision",
			"internal/vqaserve/prefill.go BindPrefillBranch",
		},
		calls: []string{"MemAlloc", "Upload"},
	},
	{
		name: "device-scope allocation chains",
		functions: []string{
			"internal/devicemath/attention_cuda_windows.go deviceGEMM",
			"internal/devicemath/gated_delta_net_cuda_windows.go GatedDeltaNetBackwardDevice",
			"internal/devicemath/gated_mlp_resident_cuda_windows.go gatedMLPBackwardTW",
			"internal/devicemath/l2norm_cuda_windows.go L2NormBackwardDevice",
			"internal/devicemath/linear_cuda_windows.go linearBackwardTW",
			"internal/devicemath/linear_cuda_windows.go linearForwardTW",
			"internal/devicemath/mha_forward_resident_cuda_windows.go MultiHeadAttentionForwardResident",
			"internal/devicemath/mha_resident_cuda_windows.go MultiHeadAttentionBackwardResident",
			"internal/devicemath/rmsnorm_cuda_windows.go RMSNormBackward",
			"internal/devicemath/rmsnorm_cuda_windows.go RMSNormForward",
			"internal/devicemath/rope_cuda_windows.go applyRoPEHalf",
			"internal/devicemath/short_conv_cuda_windows.go ShortConvBackwardDevice",
			"internal/devicemath/silu_cuda_windows.go SiLUGateForward",
			"internal/devicemath/softmax_cuda_windows.go SoftmaxBackward",
		},
		calls: []string{"upload", "alloc", "allocRaw"},
	},
	{
		name: "device-scope kernel loads",
		functions: []string{
			"internal/devicemath/gated_mlp_resident_cuda_windows.go gatedMLPBackwardTW",
			"internal/devicemath/mha_resident_cuda_windows.go MultiHeadAttentionBackwardResident",
			"internal/devicemath/silu_cuda_windows.go SiLUGateForward",
		},
		calls: []string{"function"},
	},
}

// consolidatedSurfaceGroups are the groups the plan's step shape-reading/repeats-surface consolidated in the inference surface and the answer scope.
//   - the two RoPE pairings -> rope, which the operation tells which elements pair
//   - four-axis reads of one shape -> dims4
//   - a suite's identity, dataset and split -> suiteIdentities
//   - compileSuite's decode, compile and bind of a suite -> compileKind
var consolidatedSurfaceGroups = []consolidatedGroup{
	{
		name:      "RoPE pairings",
		functions: []string{"internal/tensor/reference/ops_rope.go"},
		calls:     []string{"ropeFrequencyFactors", "ropeCosSin"},
	},
	{
		name: "four-axis reads of one shape",
		functions: []string{
			"internal/tensor/reference/ops_recurrent.go gatedDeltaNet",
			"internal/tensor/reference/ops_recurrent.go gatedLinearAttention",
			"internal/tensor/reference/ops_recurrent.go ssmScan",
			"internal/tensor/reference/ops_recurrent.go wkv6",
			"internal/tensor/reference/ops_recurrent.go wkv7",
			"internal/tensor/reference/ops_rope.go rope",
			"internal/tensor/reference/ops_rope.go ropeMulti",
		},
		calls: []string{"int"},
	},
	{
		name: "suite identity, dataset and split",
		functions: []string{
			"internal/evaluation/carried_continuation.go compileCarriedContinuation",
			"internal/evaluation/description_matching.go compileDescriptionMatching",
			"internal/evaluation/exact.go CompileExact",
			"internal/evaluation/function_call.go compileFunctionCall",
			"internal/evaluation/generated_answer.go CompileGeneratedAnswer",
			"internal/evaluation/image_retrieval.go compileImageRetrieval",
			"internal/evaluation/instruction_rules.go compileInstructionRules",
			"internal/evaluation/multiple_choice.go CompileMultipleChoice",
			"internal/evaluation/probability_mass.go compileProbabilityMass",
			"internal/evaluation/reconstruction.go compileReconstruction",
			"internal/evaluation/sequence_scoring.go compileSequenceScoring",
			"internal/evaluation/series_forecast.go compileSeriesForecast",
			"internal/evaluation/speech_round_trip.go compileSpeechRoundTrip",
			"internal/evaluation/structured_generated.go compileStructuredGenerated",
		},
		calls: []string{"JSONID"},
	},
	{
		name:      "suite decode, compile and bind",
		functions: []string{"internal/evaluation/suite.go compileSuite"},
		cases: []string{
			`""`, "GeneratedAnswerKind", "MultipleChoiceKind", "carriedContinuationKind", "descriptionMatchingKind", "functionCallKind", "imageRetrievalKind",
			"instructionRulesKind", "probabilityMassKind", "reconstructionKind", "sequenceScoringKind", "seriesForecastKind", "speechRoundTripKind",
		},
		calls: []string{"DecodeBytes"},
	},
}

// caseFixture plants a repeat inside one function's switch.
//   - clauses "a" and "b" decode, count and sum in five statements of one shape
//   - clause "c" holds one statement, so a group of "a" and "c" is a near miss
var caseFixture = map[string]string{
	"internal/suites/suites.go": `package suites

func decode(data []byte, into *[]byte) error { return nil }

func count(data []byte) (int, error) { return 0, nil }

func sum(count int) (int, error) { return 0, nil }

func Compile(kind string, data []byte) (int, error) {
	switch kind {
	case "a":
		var a []byte
		if err := decode(data, &a); err != nil {
			return 0, err
		}
		n, err := count(a)
		if err != nil {
			return 0, err
		}
		return sum(n)
	case "b":
		var b []byte
		if err := decode(data, &b); err != nil {
			return 0, err
		}
		m, err := count(b)
		if err != nil {
			return 0, err
		}
		return sum(m)
	case "c":
		return sum(0)
	}
	return 0, nil
}
`,
}

// consolidatedRepeats lists each finding of a snapshot's shape reading that repeats one of groups, and each function or case of a group the snapshot does not declare.
//   - a place serves a group <=> its file and function, or its file alone, are among the group's functions and its lines call one of the group's calls
//   - a group with cases is read over its clauses alone, each clause a function of its own, so its repeat inside one function is read as a repeat across functions is
//   - a repeated run repeats a group <=> the places that serve the group lie in runFunctions or more functions
//   - a chain repeats a group <=> its place serves the group
//   - an undeclared function or case is listed because a group none of whose places exist is checked against nothing
func consolidatedRepeats(snapshot gosyntax.Snapshot, groups []consolidatedGroup) ([]string, error) {
	reading, err := readShapes(snapshot)
	if err != nil {
		return nil, err
	}
	files, err := shapeFiles(snapshot)
	if err != nil {
		return nil, err
	}
	byPath, declared := map[string]*reachFile{}, map[string]*ast.FuncDecl{}
	for _, file := range files {
		byPath[file.source.Path] = file
		for _, declaration := range file.syntax.Decls {
			if function, isFunction := declaration.(*ast.FuncDecl); isFunction {
				name := function.Name.Name
				if function.Recv != nil {
					name = receiverType(function) + "." + name
				}
				declared[file.source.Path+" "+name] = function
			}
		}
	}
	calls := func(place shapePlace) map[string]bool {
		file, called := byPath[place.file], map[string]bool{}
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			call, isCall := node.(*ast.CallExpr)
			if !isCall || file.source.Line(call.Pos()) < place.from || file.source.Line(call.Pos()) > place.to {
				return true
			}
			switch callee := call.Fun.(type) {
			case *ast.Ident:
				called[callee.Name] = true
			case *ast.SelectorExpr:
				called[callee.Sel.Name] = true
			}
			return true
		})
		return called
	}
	var repeats []string
	for _, group := range groups {
		for _, function := range group.functions {
			if byPath[function] == nil && declared[function] == nil {
				repeats = append(repeats, fmt.Sprintf("group %q names %s, which no production file declares", group.name, function))
			}
		}
		runs, chains, held := reading.runs, reading.chains, func(place shapePlace) bool {
			return slices.Contains(group.functions, place.file) || slices.Contains(group.functions, place.file+" "+place.function)
		}
		if len(group.cases) > 0 {
			lists, missing := caseLists(byPath, declared, group)
			repeats = append(repeats, missing...)
			runs, _ = repeatedRuns(lists)
			chains, held = callChains(lists), func(shapePlace) bool { return true }
		}
		serves := func(place shapePlace) bool {
			called := calls(place)
			return held(place) && slices.ContainsFunc(group.calls, func(name string) bool { return called[name] })
		}
		for _, run := range runs {
			functions := map[string]bool{}
			for _, place := range run.places {
				if serves(place) {
					functions[place.file+" "+place.function] = true
				}
			}
			if len(functions) >= runFunctions {
				repeats = append(repeats, fmt.Sprintf("group %q repeats in %s", group.name, run))
			}
		}
		for _, chain := range chains {
			if serves(chain.place) {
				repeats = append(repeats, fmt.Sprintf("group %q repeats in %s", group.name, chain))
			}
		}
	}
	return repeats, nil
}

// caseLists reads each clause of a group's functions that names one of the group's cases as a statement list of its own, named by its function and its first case.
//   - missing = a finding for each of the group's cases no such clause names
func caseLists(byPath map[string]*reachFile, declared map[string]*ast.FuncDecl, group consolidatedGroup) ([]*statementList, []string) {
	var lists []*statementList
	shapes, named := map[string]int{}, map[string]bool{}
	for _, function := range group.functions {
		path, name, _ := strings.Cut(function, " ")
		if declared[function] == nil || declared[function].Body == nil {
			continue
		}
		ast.Inspect(declared[function].Body, func(node ast.Node) bool {
			clause, isClause := node.(*ast.CaseClause)
			if isClause && slices.ContainsFunc(clause.List, func(expression ast.Expr) bool { return slices.Contains(group.cases, types.ExprString(expression)) }) {
				for _, expression := range clause.List {
					named[types.ExprString(expression)] = true
				}
				lists = append(lists, newStatementList(byPath[path], name+" case "+types.ExprString(clause.List[0]), clause.Body, shapes))
			}
			return true
		})
	}
	var missing []string
	for _, expression := range group.cases {
		if !named[expression] {
			missing = append(missing, fmt.Sprintf("group %q names case %s, which no clause of its functions holds", group.name, expression))
		}
	}
	return lists, missing
}

// fixtureSnapshot returns a snapshot that holds the files of fixtures, each keyed by its path.
func fixtureSnapshot(t *testing.T, fixtures ...map[string]string) gosyntax.Snapshot {
	t.Helper()
	contents := map[string][]byte{}
	for _, fixture := range fixtures {
		for name, source := range fixture {
			contents[name] = []byte(source)
		}
	}
	snapshot, err := gosyntax.Snapshot{}.Overlay(contents)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

// holdConsolidated holds planted groups to the repeats want in a fixture, then groups to no repeat in the repository.
func holdConsolidated(t *testing.T, fixture gosyntax.Snapshot, planted []consolidatedGroup, want []string, groups []consolidatedGroup) {
	t.Helper()
	repeats, err := consolidatedRepeats(fixture, planted)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(repeats, want) {
		t.Errorf("the fixture's repeats =\n%s\nwant\n%s", strings.Join(repeats, "\n"), strings.Join(want, "\n"))
	}

	snapshot, err := gosyntax.DiscoverGo(filepath.Join("..", ".."), "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	repeats, err = consolidatedRepeats(snapshot, groups)
	if err != nil {
		t.Fatal(err)
	}
	if len(repeats) > 0 {
		t.Errorf("a consolidated group repeats again, want its places served by the common component:\n%s", strings.Join(repeats, "\n"))
	}
}

// TestConsolidatedGroupsNoLongerRepeat holds each group in consolidatedGroups to no repeat in the shape reading of the repository.
//   - fixture = shapeFixture with a planted run, a planted chain, and a near miss whose places call none of its calls
//   - a group whose functions the repository no longer declares is refused, because a check of nothing earns no pass
func TestConsolidatedGroupsNoLongerRepeat(t *testing.T) {
	t.Parallel()
	planted := []consolidatedGroup{
		{name: "planted run", functions: []string{"internal/store/store.go Import", "internal/archive/archive.go Restore"}, calls: []string{"load", "fetch"}},
		{name: "planted chain", functions: []string{"internal/device/device.go runner.Upload"}, calls: []string{"upload"}},
		{name: "near miss", functions: []string{"internal/store/store.go Import", "internal/archive/archive.go Restore"}, calls: []string{"launch"}},
	}
	holdConsolidated(t, fixtureSnapshot(t, shapeFixture), planted, []string{
		`group "planted run" repeats in run call-and-test lines=20 places=2 internal/archive/archive.go:10-19 Restore, internal/store/store.go:10-19 Import`,
		`group "planted chain" repeats in chain receiver=device pairs=3 lines=12 internal/device/device.go:12-23 runner.Upload`,
	}, consolidatedGroups)
}

// TestConsolidatedSurfaceGroupsNoLongerRepeat holds each group in consolidatedSurfaceGroups to no repeat in the shape reading of the repository.
//   - fixture = caseFixture with a planted repeat inside one switch, a near miss of a clause of one statement, and a case no clause names
//   - fixture = shapeFixture with a planted run named by its files alone
//   - a group whose functions or cases the repository no longer declares is refused, because a check of nothing earns no pass
func TestConsolidatedSurfaceGroupsNoLongerRepeat(t *testing.T) {
	t.Parallel()
	compile := []string{"internal/suites/suites.go Compile"}
	planted := []consolidatedGroup{
		{name: "planted cases", functions: compile, cases: []string{`"a"`, `"b"`}, calls: []string{"decode"}},
		{name: "near miss", functions: compile, cases: []string{`"a"`, `"c"`}, calls: []string{"decode"}},
		{name: "absent case", functions: compile, cases: []string{`"d"`}, calls: []string{"decode"}},
		{name: "planted files", functions: []string{"internal/store/store.go", "internal/archive/archive.go"}, calls: []string{"load", "fetch"}},
	}
	holdConsolidated(t, fixtureSnapshot(t, shapeFixture, caseFixture), planted, []string{
		`group "planted cases" repeats in run call-and-test lines=18 places=2 internal/suites/suites.go:12-20 Compile case "a", internal/suites/suites.go:22-30 Compile case "b"`,
		`group "absent case" names case "d", which no clause of its functions holds`,
		`group "planted files" repeats in run call-and-test lines=20 places=2 internal/archive/archive.go:10-19 Restore, internal/store/store.go:10-19 Import`,
	}, consolidatedSurfaceGroups)
}
