package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/export"
	"overgo/internal/gguf"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

// sizePreview is a GGUF workspace served by a handler as the Export and Model
// Builder tabs reach it, writing under root. It catalogues three models:
// hermetic.gguf at base, tuned.gguf, its fine-tuned copy, and padded.gguf, the
// hermetic model declaring one more tokenizer key, so a merge of either of the
// others onto it is a merge of another tokenizer, which the writer refuses.
type sizePreview struct {
	root, base string
	workspace  *GGUFWorkspace
	handler    *Handler
}

// unsizedWriters are the launcher's writers as they were before a size could
// be asked, which is what every other test of the workspace still passes.
func unsizedWriters() GGUFWriters {
	return GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF}
}

// sizedWriters are the writers the launcher passes now, each with the size of
// what it would write.
func sizedWriters() GGUFWriters {
	writers := unsizedWriters()
	writers.ExportSize, writers.MergeSize = export.GGUFSize, export.MergeGGUFSize
	return writers
}

// newSizePreview serves the workspace as a member of a workspace set, which is
// how the launcher serves it, or as the handler's whole workspace, which is how
// the other tests of this package do.
func newSizePreview(t *testing.T, writers GGUFWriters, inSet bool) sizePreview {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	base, _ := mergeFixture(t, store)
	padded := filepath.Join(t.TempDir(), "padded.gguf")
	if err := os.Rename(testutil.HermeticLlamaGGUFWith(t, 64, testutil.GGUFScalar("tokenizer.ggml.padding_token_id", gguf.ValueTypeUint32, uint32(0))), padded); err != nil {
		t.Fatal(err)
	}
	if err := servingtest.PublishActiveGGUFWithPolicy(t.Context(), store, padded, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	workspace := NewGGUFWorkspace(store, root, 64, writers, "")
	var served WorkflowWorkspaceAPI = workspace
	if inSet {
		served = WorkflowWorkspaceSet{workspace}
	}
	handler, err := New(Config{Repository: store, Workflows: served}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	return sizePreview{root: root, base: base, workspace: workspace, handler: handler}
}

// capability is the one capability of a tab that match picks.
func (preview sizePreview) capability(t *testing.T, kind WorkflowKind, match func(WorkflowCapability) bool) WorkflowCapability {
	t.Helper()
	capabilities, err := preview.workspace.WorkflowCapabilities(t.Context(), kind)
	if err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(capabilities, match)
	if index < 0 {
		t.Fatalf("%s lists no such capability among %+v", kind, capabilities)
	}
	return capabilities[index]
}

// on picks the capability on the catalogued model of that name.
func on(name string) func(WorkflowCapability) bool {
	return func(capability WorkflowCapability) bool { return capability.Name == name }
}

// sizeAnswer is what a command route answered: its status and text, and the
// size in bytes when the answer carries one.
type sizeAnswer struct {
	status int
	text   string
	bytes  uint64
	sized  bool
}

// ask posts the submission the tab's form would post to its command route. An
// admitted submission must still answer its commands, with or without a size.
func (preview sizePreview) ask(t *testing.T, kind WorkflowKind, capability WorkflowCapability, input string) sizeAnswer {
	t.Helper()
	submission := `{"task":"` + string(capability.Task) + `","recipe":"` + capability.Recipe.String() + `","input":` + input + `}`
	answered := serveTestRequest(preview.handler, http.MethodPost, "/"+string(kind)+"/command", submission)
	answer := sizeAnswer{status: answered.Code, text: answered.Body.String()}
	if answer.status != http.StatusOK {
		return answer
	}
	var body struct {
		Commands []workflowCommand `json:"commands"`
		Bytes    *uint64           `json:"bytes"`
	}
	if err := json.Unmarshal(answered.Body.Bytes(), &body); err != nil || len(body.Commands) == 0 {
		t.Fatalf("%s command for %s answered %s: %v", kind, input, answer.text, err)
	}
	if body.Bytes != nil {
		answer.bytes, answer.sized = *body.Bytes, true
	}
	return answer
}

// run runs the submission as the tab's Run button would have the workspace
// run it, and answers the run's refusal.
func (preview sizePreview) run(t *testing.T, kind WorkflowKind, capability WorkflowCapability, input string) error {
	t.Helper()
	_, err := preview.workspace.ExecuteWorkflow(t.Context(), kind, capability, json.RawMessage(input), &recordingReporter{})
	return err
}

// writtenBytes is the size of a file a run wrote.
func writtenBytes(t *testing.T, elements ...string) uint64 {
	t.Helper()
	info, err := os.Stat(filepath.Join(elements...))
	if err != nil {
		t.Fatal(err)
	}
	return uint64(info.Size())
}

// TestExportSizePreviewAnswersTheBytesAnExportWrites holds the size the Export
// tab's command route answers to the file a run of the same submission then
// writes, for every type the tab offers, with the workspace served from a set
// as the launcher serves it: an answered size is that file's size to the byte
// and asking wrote nothing, and a type the run refuses is refused when asked.
// Two types of different storage answer different sizes, neither the source's,
// so no constant and no echo of the source passes.
func TestExportSizePreviewAnswersTheBytesAnExportWrites(t *testing.T) {
	t.Parallel()
	preview := newSizePreview(t, sizedWriters(), true)
	capability := preview.capability(t, workflowExport, on("hermetic.gguf"))
	sizes := map[string]uint64{}
	for _, name := range capability.Controls[0].Choices {
		input := `{"type":"` + name + `","output":"size-` + name + `"}`
		answer := preview.ask(t, workflowExport, capability, input)
		directory := filepath.Join(preview.root, "size-"+name)
		if _, err := os.Stat(directory); !os.IsNotExist(err) {
			t.Fatalf("asking the size of a %s export left %s: %v", name, directory, err)
		}
		runErr := preview.run(t, workflowExport, capability, input)
		switch {
		case answer.status == http.StatusBadRequest && runErr != nil:
			// Refused when asked, as the run refuses it.
		case answer.status == http.StatusOK && answer.sized && runErr == nil:
			if wrote := writtenBytes(t, directory, "hermetic."+name+".gguf"); wrote != answer.bytes {
				t.Fatalf("a %s export was sized at %d bytes and wrote %d", name, answer.bytes, wrote)
			}
			sizes[name] = answer.bytes
		default:
			t.Fatalf("a %s export answered %+v and its run answered %v; the two must agree", name, answer, runErr)
		}
	}
	source := writtenBytes(t, preview.base)
	if f16, q8 := sizes["f16"], sizes["q8_0"]; f16 == 0 || q8 == 0 || f16 == q8 || f16 == source {
		t.Fatalf("f16 was sized at %d bytes and q8_0 at %d, from a source of %d: each must be answered, and they must differ from each other and f16 from the source", f16, q8, source)
	}
	if answer := preview.ask(t, workflowExport, capability, `{"type":"iq2_xxs","output":"importance"}`); answer.status != http.StatusBadRequest {
		t.Fatalf("a type that needs an importance matrix answered %+v; the run refuses it, so asking must", answer)
	}
}

// TestExportSizePreviewAnswersTheBytesAMergeWrites holds the size Model
// Builder's command route answers for a merge to the merged file a run of the
// same submission writes. What the run refuses is refused when asked, for the
// run's reason: a fine-tuned model the catalog does not hold, a scale the run
// cannot read, and a fine-tuned model of another tokenizer than its base, which
// only the chosen model's own file can show. The adapter fold, which no run
// writes, answers its commands and no size.
func TestExportSizePreviewAnswersTheBytesAMergeWrites(t *testing.T) {
	t.Parallel()
	preview := newSizePreview(t, sizedWriters(), false)
	merge := preview.capability(t, workflowModelBuild, on("hermetic.gguf"))
	input := `{"fine_tuned":"tuned.gguf","scale":0.5,"output":"merged"}`
	answer := preview.ask(t, workflowModelBuild, merge, input)
	if answer.status != http.StatusOK || !answer.sized {
		t.Fatalf("a merge answered %+v, want its size", answer)
	}
	if _, err := os.Stat(filepath.Join(preview.root, "merged")); !os.IsNotExist(err) {
		t.Fatalf("asking the size of a merge left its directory: %v", err)
	}
	if err := preview.run(t, workflowModelBuild, merge, input); err != nil {
		t.Fatal(err)
	}
	if wrote, source := writtenBytes(t, preview.root, "merged", "hermetic.merged.gguf"), writtenBytes(t, preview.base); wrote != answer.bytes || source == answer.bytes {
		t.Fatalf("a merge was sized at %d bytes and wrote %d, onto a base of %d: the size must be the written file's, and no echo of the base", answer.bytes, wrote, source)
	}
	for _, refused := range []struct {
		capability     WorkflowCapability
		input, refusal string
	}{
		{merge, `{"fine_tuned":"absent.gguf","scale":0.5,"output":"other"}`, `0 catalogued models are named`},
		{merge, `{"fine_tuned":"tuned.gguf","scale":1e39,"output":"other"}`, `float32`},
		{preview.capability(t, workflowModelBuild, on("padded.gguf")), `{"fine_tuned":"tuned.gguf","scale":0.5,"output":"other"}`, `tokenizer.ggml.padding_token_id`},
	} {
		if err := preview.run(t, workflowModelBuild, refused.capability, refused.input); err == nil || !strings.Contains(err.Error(), refused.refusal) {
			t.Fatalf("a run of the merge %s onto %s answered %v, want it refused naming %q: without that it is no refusal to hold the size to", refused.input, refused.capability.Name, err, refused.refusal)
		}
		if answer := preview.ask(t, workflowModelBuild, refused.capability, refused.input); answer.status != http.StatusBadRequest || !strings.Contains(answer.text, refused.refusal) {
			t.Fatalf("the merge %s onto %s answered %+v; the run refuses it, so asking must, naming %q", refused.input, refused.capability.Name, answer, refused.refusal)
		}
	}
	fold := preview.capability(t, workflowModelBuild, func(capability WorkflowCapability) bool { return capability.Refusal != "" })
	if answer := preview.ask(t, workflowModelBuild, fold, `{}`); answer.status != http.StatusOK || answer.sized {
		t.Fatalf("the adapter fold answered %+v; it must answer its commands and no size", answer)
	}
}

// TestExportSizePreviewIsAbsentWithoutSizeFunctions holds a workspace whose
// launcher gave it writers and no size functions to the answer it gave before
// sizes existed: the commands, and no size, for an export and for a merge.
func TestExportSizePreviewIsAbsentWithoutSizeFunctions(t *testing.T) {
	t.Parallel()
	preview := newSizePreview(t, unsizedWriters(), false)
	for kind, input := range map[WorkflowKind]string{
		workflowExport:     `{"type":"f16","output":"unsized"}`,
		workflowModelBuild: `{"fine_tuned":"tuned.gguf","scale":0.5,"output":"unsized"}`,
	} {
		if answer := preview.ask(t, kind, preview.capability(t, kind, on("hermetic.gguf")), input); answer.status != http.StatusOK || answer.sized {
			t.Fatalf("%s without a size function answered %+v, want its commands and no size", kind, answer)
		}
	}
}
