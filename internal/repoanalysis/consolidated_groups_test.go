package repoanalysis

import (
	"fmt"
	"go/ast"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// consolidatedGroup is one group of repeated logic that a common component replaced.
//   - functions = the file and the function of each place the group held, written as shapePlace writes them without its lines
//   - calls = the names the repeated logic called and the common component now calls in one place
type consolidatedGroup struct {
	name      string
	functions []string
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

// consolidatedRepeats lists each finding of a snapshot's shape reading that repeats one of groups, and each function of a group the snapshot does not declare.
//   - a place serves a group <=> its file and function are among the group's functions and its lines call one of the group's calls
//   - a repeated run repeats a group <=> the places that serve the group lie in runFunctions or more functions
//   - a chain repeats a group <=> its place serves the group
//   - an undeclared function is listed because a group none of whose places exist is checked against nothing
func consolidatedRepeats(snapshot gosyntax.Snapshot, groups []consolidatedGroup) ([]string, error) {
	reading, err := readShapes(snapshot)
	if err != nil {
		return nil, err
	}
	files, err := shapeFiles(snapshot)
	if err != nil {
		return nil, err
	}
	byPath, declared := map[string]*reachFile{}, map[string]bool{}
	for _, file := range files {
		byPath[file.source.Path] = file
		for _, declaration := range file.syntax.Decls {
			if function, isFunction := declaration.(*ast.FuncDecl); isFunction {
				name := function.Name.Name
				if function.Recv != nil {
					name = receiverType(function) + "." + name
				}
				declared[file.source.Path+" "+name] = true
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
		serves := func(place shapePlace) bool {
			called := calls(place)
			return slices.Contains(group.functions, place.file+" "+place.function) && slices.ContainsFunc(group.calls, func(name string) bool { return called[name] })
		}
		for _, function := range group.functions {
			if !declared[function] {
				repeats = append(repeats, fmt.Sprintf("group %q names %s, which no production file declares", group.name, function))
			}
		}
		for _, run := range reading.runs {
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
		for _, chain := range reading.chains {
			if serves(chain.place) {
				repeats = append(repeats, fmt.Sprintf("group %q repeats in %s", group.name, chain))
			}
		}
	}
	return repeats, nil
}

// TestConsolidatedGroupsNoLongerRepeat holds each group in consolidatedGroups to no repeat in the shape reading of the repository.
//   - fixture = shapeFixture with a planted run, a planted chain, and a near miss whose places call none of its calls
//   - a group whose functions the repository no longer declares is refused, because a check of nothing earns no pass
func TestConsolidatedGroupsNoLongerRepeat(t *testing.T) {
	t.Parallel()
	contents := map[string][]byte{}
	for name, source := range shapeFixture {
		contents[name] = []byte(source)
	}
	snapshot, err := gosyntax.Snapshot{}.Overlay(contents)
	if err != nil {
		t.Fatal(err)
	}
	planted := []consolidatedGroup{
		{name: "planted run", functions: []string{"internal/store/store.go Import", "internal/archive/archive.go Restore"}, calls: []string{"load", "fetch"}},
		{name: "planted chain", functions: []string{"internal/device/device.go runner.Upload"}, calls: []string{"upload"}},
		{name: "near miss", functions: []string{"internal/store/store.go Import", "internal/archive/archive.go Restore"}, calls: []string{"launch"}},
	}
	repeats, err := consolidatedRepeats(snapshot, planted)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		`group "planted run" repeats in run call-and-test lines=20 places=2 internal/archive/archive.go:10-19 Restore, internal/store/store.go:10-19 Import`,
		`group "planted chain" repeats in chain receiver=device pairs=3 lines=12 internal/device/device.go:12-23 runner.Upload`,
	}
	if !slices.Equal(repeats, want) {
		t.Errorf("the fixture's repeats =\n%s\nwant\n%s", strings.Join(repeats, "\n"), strings.Join(want, "\n"))
	}

	snapshot, err = gosyntax.DiscoverGo(filepath.Join("..", ".."), "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	repeats, err = consolidatedRepeats(snapshot, consolidatedGroups)
	if err != nil {
		t.Fatal(err)
	}
	if len(repeats) > 0 {
		t.Errorf("a consolidated group repeats again, want its places served by the common component:\n%s", strings.Join(repeats, "\n"))
	}
}
