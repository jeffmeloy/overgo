package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestStructuredMediaReportProjection(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	storePath := retainedReferenceStore(roots.Store)
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	t.Run("shared retained projection", func(t *testing.T) {
		assertStructuredMediaReport(t, root, storePath)
	})
	t.Run("protocol counterexamples", func(t *testing.T) {
		assertStructuredMediaProtocol(t, root, store)
	})
}

func assertStructuredMediaReport(t *testing.T, root, storePath string) {
	t.Helper()
	output, err := generateMediaReport(root, storePath)
	if err != nil {
		t.Fatal(err)
	}
	var value mediaProjection
	if err := json.Unmarshal(output.JSON, &value); err != nil {
		t.Fatal(err)
	}
	var index mediaValidationIndex
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaValidationPath), &index); err != nil {
		t.Fatal(err)
	}
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		t.Fatal(err)
	}
	if value.Validation == nil || value.Protocol == nil || len(value.Validation.Checks) != len(index.Checks) || len(value.Protocol.Cases) != len(protocol.Cases) || value.Counts.Activations != len(value.Rows) {
		t.Fatal("declared denominator omitted or totals disagree")
	}
	if value.Validation.Passed == 0 || value.Validation.Unresolved == 0 || value.Validation.Passed+value.Validation.Unresolved != len(index.Checks) {
		t.Fatal("historical successes or failures lost", value.Validation)
	}
	for i, check := range value.Validation.Checks {
		declared := index.Checks[i]
		if check.Evidence != declared.Evidence || check.PriorEvidence != declared.PriorEvidence || check.Scope != declared.Scope || check.Source != cmp.Or(declared.Source, index.Source) ||
			!maps.Equal(check.SourceFiles, declared.SourceFiles) || !maps.Equal(check.Harness, declared.Harness) || len(check.Assertions) == 0 || check.RepairOwner == "" {
			t.Fatal("lost assertion scope, provenance or repair ownership", check.Name)
		}
	}
	if value.Validation.PriorCapacity != index.PriorCapacity {
		t.Fatal("prior capacity checkpoint lost")
	}
	for i, cell := range value.Protocol.Cases {
		if cell.ID != protocol.Cases[i].ID || cell.Model != protocol.Cases[i].Model || cell.Task != protocol.Cases[i].Task || cell.Run != protocol.Cases[i].Run || !slices.Equal(cell.Outputs, protocol.Cases[i].Outputs) || cell.Gap == "" || cell.RepairOwner == "" {
			t.Fatal("lost protocol scope or inferred current acceptance", cell.ID)
		}
	}
	for _, row := range value.Rows {
		if row.GenerationGap == "" || row.RepairOwner == "" {
			t.Fatal("activation-only row gained generation acceptance")
		}
	}
	var rendered bytes.Buffer
	writeMediaValidation(&rendered, value.Validation)
	writeMediaProtocol(&rendered, value.Protocol)
	if !bytes.Contains(output.Markdown, rendered.Bytes()) {
		t.Fatal("human report differs from typed evidence projection")
	}
	t.Logf("%d activations; %d declared checks (%d historical passes, %d unresolved); %d protocol cases; no acquisition or current-source promotion", len(value.Rows), len(index.Checks), value.Validation.Passed, value.Validation.Unresolved, len(protocol.Cases))
}

func assertStructuredMediaProtocol(t *testing.T, root string, reader artifact.Reader) {
	t.Helper()
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		t.Fatal(err)
	}
	write := func(value imageVideoProtocol) string {
		t.Helper()
		fixture := t.TempDir()
		path := filepath.Join(fixture, imageVideoProtocolPath)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		return fixture
	}
	for name, mutate := range map[string]func(*imageVideoProtocol){
		"empty denominator":  func(p *imageVideoProtocol) { p.Cases = nil },
		"duplicate case":     func(p *imageVideoProtocol) { p.Cases = append(p.Cases, p.Cases[0]) },
		"missing assertions": func(p *imageVideoProtocol) { p.RequiredChecks = nil },
		"duplicate assertion": func(p *imageVideoProtocol) {
			p.RequiredChecks = append(slices.Clone(p.RequiredChecks), p.RequiredChecks[0])
		},
		"task scope":          func(p *imageVideoProtocol) { p.Cases[0].Task = recipe.TaskInference },
		"negative width":      func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Width = -1 },
		"zero height":         func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Height = 0 },
		"empty frames":        func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Frames = 0 },
		"negative frame rate": func(p *imageVideoProtocol) { p.Cases[0].Observations[0].FPS = -1 },
		"negative delay":      func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Delay = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := protocol
			changed.Cases = slices.Clone(protocol.Cases)
			changed.Cases[0].Observations = slices.Clone(protocol.Cases[0].Observations)
			mutate(&changed)
			if _, err := projectMediaProtocol(t.Context(), write(changed), reader, nil); err == nil {
				t.Fatal("invalid protocol accepted")
			}
		})
	}
	first := protocol.Cases[0]
	other := testutil.ArtifactID(t, artifact.KindRecipe, "replacement active recipe")
	rows := []mediaProjectionRow{{Model: first.Model, Task: first.Task, Recipe: other}}
	projected, err := projectMediaProtocol(t.Context(), root, reader, rows)
	if err != nil || !projected.Cases[0].Retained || projected.Cases[0].CurrentRecipe != other || projected.Cases[0].Recipe != first.Recipe || !strings.Contains(projected.Cases[0].Gap, "different recipe") {
		t.Fatal("replacement activation overwrote historical authority", err)
	}
	changed := protocol
	changed.Cases = slices.Clone(protocol.Cases)
	changed.Cases[0].Run = testutil.ArtifactID(t, artifact.KindRun, "missing retained generation")
	projected, err = projectMediaProtocol(t.Context(), write(changed), reader, rows)
	if err != nil || projected.Cases[0].Retained || projected.Cases[0].Gap == "" || projected.Retained >= len(protocol.Cases) {
		t.Fatal("missing run gained evidence credit", err)
	}
}

func TestMediaEvidenceBinding(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_capacity.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoCapacitySHA256); err != nil {
		t.Fatal(err)
	}
	var inputs struct {
		Bundle     mediaCapacityBundle
		Assertions map[string]map[string][]string
	}
	if err := json.Unmarshal(raw, &inputs.Bundle); err != nil {
		t.Fatal(err)
	}
	var assertions map[string]map[string][]string
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &assertions); err != nil {
		t.Fatal(err)
	}
	inputs.Assertions = map[string]map[string][]string{}
	for _, check := range inputs.Bundle.Checks {
		inputs.Assertions[check.Name] = assertions[check.Name]
	}
	identify := func(value any) artifact.ID {
		t.Helper()
		id, err := artifact.JSONID(artifact.KindEvidence, value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	inputID := identify(inputs)
	var reader artifact.Reader = store
	read := func(ctx context.Context, id artifact.ID) ([]byte, error) {
		content, found, err := artifact.ReadContent(ctx, reader, id)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, fmt.Errorf("media binding: missing %s", id)
		}
		return content.Data, nil
	}
	runs := 0
	checks := []automationcheck.Check{{
		Descriptor: automationcheck.Descriptor{Name: "retained-media-capacity-binding", Phase: runrecord.PhaseTest, Always: true},
		Run: func(ctx context.Context, _ automationcheck.Invocation) (bool, string, error) {
			runs++
			refs := []artifact.ID{inputs.Bundle.Native, inputs.Bundle.Host, inputs.Bundle.Acquisition}
			refs = append(refs, inputs.Bundle.Processing...)
			refs = append(refs, inputs.Bundle.Failed...)
			refs = append(refs, slices.Collect(maps.Values(inputs.Bundle.Harnesses))...)
			refs = append(refs, slices.Collect(maps.Values(inputs.Bundle.Loaders))...)
			for _, source := range inputs.Bundle.SourceFiles {
				refs = append(refs, source.Evidence)
			}
			if err := artifact.ReadContents(ctx, reader, refs, func(artifact.Content) error { return nil }); err != nil {
				return false, "", err
			}
			protocol, err := read(ctx, inputs.Bundle.Protocol)
			if err != nil {
				return false, "", err
			}
			environment, err := read(ctx, inputs.Bundle.Environment)
			if err != nil {
				return false, "", err
			}
			if _, err := checkMediaCapacityBinding(inputs.Bundle, protocol, environment); err != nil {
				return false, "", err
			}
			for _, check := range inputs.Bundle.Checks {
				receipt, err := read(ctx, check.Evidence)
				if err != nil {
					return false, "", err
				}
				if err := checkMediaTestReceipt(check.Name, receipt, inputs.Assertions[check.Name]); err != nil {
					return false, "", err
				}
			}
			return false, "Historical capacity bindings and named assertions; no current model-quality or performance promotion", nil
		},
	}}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	definition := invocations[0].ID
	manifest := testutil.ArtifactID(t, artifact.KindProfile, "media binding fixture manifest")
	bind := func(document string, input, environment artifact.ID) automationcheck.Invocation {
		t.Helper()
		plan, err := automationcheck.BindManifestPlan(manifest, manifest, input.DigestHex(), identify(document).DigestHex(), automationcheck.Surface{Identity: "retained-media-audit-fixture"}, automationcheck.Impact{}, invocations)
		if err != nil {
			t.Fatal(err)
		}
		bound, err := automationcheck.BindManifestExecution(plan, invocations[0], []artifact.ID{input}, &automationcheck.ReuseBinding{Input: input, Environment: environment})
		if err != nil {
			t.Fatal(err)
		}
		return bound
	}
	originalInvocation := bind("before unrelated documentation edit", inputID, inputs.Bundle.Environment)
	original, err := automationcheck.Run(t.Context(), originalInvocation)
	if err != nil {
		t.Fatal(err)
	}
	cache := automationcheck.NewEvidenceCache(inputs.Bundle.Environment)
	slot := originalInvocation
	slot.ID = definition
	cache.Record(slot, inputID, original)
	encoded, err := json.Marshal(cache)
	if err != nil {
		t.Fatal(err)
	}
	var resumed automationcheck.EvidenceCache
	if err := json.Unmarshal(encoded, &resumed); err != nil {
		t.Fatal(err)
	}
	successor := bind("after unrelated documentation edit", inputID, inputs.Bundle.Environment)
	successor.ID = definition
	reused, found := resumed.Lookup(successor, inputID)
	if !found || runs != 1 || reused.Source == nil || reused.Source.Evidence != original.ID || reused.ID == original.ID {
		t.Fatal("exact audit was reacquired or lost original authority after restart")
	}
	if err := automationcheck.ValidateReuseAuthority(reused, definition); err != nil {
		t.Fatal(err)
	}
	if err := original.VerifyIdentity(); err != nil {
		t.Fatal(err)
	}
	t.Run("changed assertion contract", func(t *testing.T) {
		changed := inputs
		changed.Assertions = maps.Clone(inputs.Assertions)
		changed.Assertions[inputs.Bundle.Checks[0].Name] = map[string][]string{"overgo/unrelated": {"TestUnrelated"}}
		id := identify(changed)
		candidate := bind("changed assertion contract", id, inputs.Bundle.Environment)
		candidate.ID = definition
		if _, found := resumed.Lookup(candidate, id); found {
			t.Fatal("another package and assertion contract reused prior acceptance")
		}
	})
	t.Run("missing native fixture", func(t *testing.T) {
		reader = verificationContentFaultReader{Reader: store, id: inputs.Bundle.Native, absent: true}
		defer func() { reader = store }()
		if _, err := automationcheck.Run(t.Context(), originalInvocation); err == nil {
			t.Fatal("missing native fixture gained audit credit")
		}
	})
	for _, test := range []struct {
		name   string
		change func(*mediaCapacityBundle)
	}{
		{"source", func(b *mediaCapacityBundle) { b.Source = strings.Repeat("b", len(b.Source)) }},
		{"protocol", func(b *mediaCapacityBundle) { b.Protocol = identify("changed protocol") }},
		{"environment", func(b *mediaCapacityBundle) { b.Environment = identify("changed environment") }},
		{"oracle", func(b *mediaCapacityBundle) { b.NativeOracle = identify("changed oracle").DigestHex() }},
		{"artifact", func(b *mediaCapacityBundle) { b.Native = identify("changed native artifact") }},
		{"overlay", func(b *mediaCapacityBundle) {
			b.Harnesses = map[string]artifact.ID{"changed overlay": identify("changed overlay")}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := inputs
			test.change(&changed.Bundle)
			id := identify(changed)
			candidate := bind(test.name, id, changed.Bundle.Environment)
			candidate.ID = definition
			if _, found := resumed.Lookup(candidate, id); found {
				t.Fatal("changed media inputs reused prior acceptance")
			}
			forged := reused
			forged.Authority = candidate.Authority
			id, err := forged.Identity()
			if err != nil {
				t.Fatal(err)
			}
			forged.ID = id
			if automationcheck.ValidateReuseAuthority(forged, definition) == nil {
				t.Fatal("reidentified successor forged original input authority")
			}
		})
	}
	t.Logf("%d retained receipt contracts audited once; exact restart reuse accepted; changed binding and forged successor refused; historical scope retained", len(inputs.Bundle.Checks))
}

type verificationContentFaultReader struct {
	artifact.Reader
	id     artifact.ID
	absent bool
}

func (r verificationContentFaultReader) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if id == r.id {
		if r.absent {
			return artifact.Descriptor{}, nil, false, nil
		}
		descriptor, found, err := r.Reader.Artifact(ctx, id)
		return descriptor, bytes.NewReader(bytes.Repeat([]byte{'?'}, int(descriptor.Size))), found, err
	}
	return r.Reader.OpenContent(ctx, id)
}

func TestMediaReceiptProvenance(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "producer")
	store, err := overgodb.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	contract := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "text/plain", Schema: "overgo/source/v1"}
	content, err := contract.ContentBytes([]byte("package fixture\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/media-overlay", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	relocated := filepath.Join(root, "consumer")
	if err := os.Rename(path, relocated); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.OpenReadOnly(relocated)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	missing, err := contract.Identify([]byte("unresolved overlay"))
	if err != nil {
		t.Fatal(err)
	}
	const producerPath = "C:/retired/worktree/fixture_test.go"
	valid := mediaValidationCheck{
		Source:      strings.Repeat("a", 40),
		SourceFiles: map[string]string{"fixture.go": content.Descriptor.ID.DigestHex()},
		Harness:     map[string]artifact.ID{producerPath: content.Descriptor.ID},
	}
	for _, test := range []struct {
		name              string
		change            func(*mediaValidationCheck)
		corrupt, accepted bool
	}{
		{name: "relocated content retains producer path", accepted: true},
		{name: "unrelated source file", change: func(*mediaValidationCheck) {
			if err := os.WriteFile(filepath.Join(root, "unrelated.txt"), []byte("unrelated change"), 0600); err != nil {
				t.Fatal(err)
			}
		}, accepted: true},
		{name: "corrupt content", corrupt: true},
		{name: "missing overlay", change: func(v *mediaValidationCheck) { v.Harness[producerPath] = missing }},
		{name: "invalid overlay identity", change: func(v *mediaValidationCheck) { v.Harness[producerPath] = artifact.ID{} }},
		{name: "empty overlay path", change: func(v *mediaValidationCheck) { v.Harness[""] = content.Descriptor.ID }},
		{name: "invalid producer", change: func(v *mediaValidationCheck) { v.Source = "unknown" }},
		{name: "invalid source hash", change: func(v *mediaValidationCheck) { v.SourceFiles["fixture.go"] = "unknown" }},
		{name: "escaping source path", change: func(v *mediaValidationCheck) { v.SourceFiles["../fixture.go"] = content.Descriptor.ID.DigestHex() }},
	} {
		t.Run(test.name, func(t *testing.T) {
			check := valid
			check.Harness, check.SourceFiles = maps.Clone(valid.Harness), maps.Clone(valid.SourceFiles)
			if test.change != nil {
				test.change(&check)
			}
			var reader artifact.Reader = store
			if test.corrupt {
				reader = verificationContentFaultReader{Reader: store, id: content.Descriptor.ID}
			}
			if err := checkMediaReceiptProvenance(t.Context(), reader, check); (err == nil) != test.accepted {
				t.Fatalf("accepted=%t want=%t: %v", err == nil, test.accepted, err)
			}
		})
	}
}

func TestMediaReceiptAssertions(t *testing.T) {
	const pkg = "overgo/fixture"
	const assertion = "TestGeneration"
	valid := "{\"Action\":\"start\",\"Package\":\"overgo/fixture\"}\n" +
		"{\"Action\":\"run\",\"Package\":\"overgo/fixture\",\"Test\":\"TestGeneration\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/fixture\",\"Test\":\"TestGeneration\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/fixture\"}\n"
	for _, test := range []struct {
		name, output string
		required     map[string][]string
		passed       bool
	}{
		{"complete", valid, map[string][]string{pkg: {assertion}}, true},
		{"empty denominator", valid, nil, false},
		{"empty package denominator", valid, map[string][]string{pkg: {}}, false},
		{"unrelated pass", valid, map[string][]string{pkg: {"TestOther"}}, false},
		{"same name wrong package", strings.ReplaceAll(valid, pkg, pkg+"other"), map[string][]string{pkg: {assertion}}, false},
		{"omitted assertion", valid, map[string][]string{pkg: {assertion, "TestOther"}}, false},
		{"renamed assertion", strings.ReplaceAll(valid, assertion, assertion+"Renamed"), map[string][]string{pkg: {assertion}}, false},
		{"skipped assertion", strings.ReplaceAll(valid, `"pass"`, `"skip"`), map[string][]string{pkg: {assertion}}, false},
		{"failed assertion", strings.ReplaceAll(valid, `"pass"`, `"fail"`), map[string][]string{pkg: {assertion}}, false},
		{"malformed stream", valid + "{bad json}\n", map[string][]string{pkg: {assertion}}, false},
		{"unfinished package", valid[:strings.LastIndex(strings.TrimSuffix(valid, "\n"), "\n")+1], map[string][]string{pkg: {assertion}}, false},
		{"terminal without acquisition", strings.Join(strings.Split(valid, "\n")[2:], "\n"), map[string][]string{pkg: {assertion}}, false},
		{"repeated lifecycle", valid + valid, map[string][]string{pkg: {assertion}}, false},
		{"duplicate assertion", valid, map[string][]string{pkg: {assertion, assertion}}, false},
		{"two packages same name", valid + strings.ReplaceAll(valid, pkg, pkg+"other"), map[string][]string{pkg: {assertion}, pkg + "other": {assertion}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := checkMediaTestReceipt(test.name, []byte(test.output), test.required); (err == nil) != test.passed {
				t.Fatalf("accepted=%v want=%v: %v", err == nil, test.passed, err)
			}
			if got, _ := mediaValidationVerdict([]byte(test.output), test.required); got != test.passed {
				t.Fatal("report and retained acceptance disagree")
			}
		})
	}
	t.Run("retained contracts", func(t *testing.T) {
		if testing.Short() {
			t.Skip(testskip.ShortIntegration)
		}
		root, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			t.Fatal(err)
		}
		roots, err := dataroot.Resolve(root)
		if err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		var bindings map[string]map[string][]string
		if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &bindings); err != nil {
			t.Fatal(err)
		}
		seen := map[string]bool{}
		accepted, rejected := 0, 0
		for _, path := range []string{mediaValidationPath, "docs/image_video_lifecycle.json", "docs/image_video_capacity.json"} {
			var index struct {
				Checks []struct {
					mediaValidationCheck
					Required []string `json:"required_tests"`
				}
			}
			if err := jsonfile.Decode(filepath.Join(root, path), &index); err != nil {
				t.Fatal(err)
			}
			for _, check := range index.Checks {
				if seen[check.Name] || len(bindings[check.Name]) == 0 {
					t.Fatal("duplicate check or missing assertion contract", check.Name)
				}
				seen[check.Name] = true
				if err := checkMediaReceiptProvenance(t.Context(), store, check.mediaValidationCheck); err != nil {
					t.Fatal(check.Name, err)
				}
				content, found, err := artifact.ReadContent(t.Context(), store, check.Evidence)
				if err != nil || !found {
					t.Fatal("retained acquisition absent", check.Name, err)
				}
				report, parseErr := testevidence.GoTestJSONReport(string(content.Data))
				err = checkMediaTestReceipt(check.Name, content.Data, bindings[check.Name])
				if parseErr != nil || testevidence.RequireComplete(report) != nil {
					if err == nil {
						t.Fatal("historical failure gained assertion credit", check.Name)
					}
					rejected++
					continue
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(check.Required) != 0 {
					requireMediaTestReceipt(t, check.Name, content.Data, check.Required, bindings)
				}
				accepted++
				substituted := maps.Clone(bindings[check.Name])
				for pkg, names := range substituted {
					delete(substituted, pkg)
					substituted[pkg+"-unrelated"] = names
					break
				}
				if checkMediaTestReceipt(check.Name, content.Data, substituted) == nil {
					t.Fatal("retained names credited to an unrelated package", check.Name)
				}
			}
		}
		if len(seen) != len(bindings) || accepted == 0 || rejected == 0 {
			t.Fatalf("contracts=%d checks=%d accepted=%d retained failures=%d", len(bindings), len(seen), accepted, rejected)
		}
		t.Logf("%d exact contracts: %d accepted, %d historical failures retained; wrong-package substitutions refused; no model execution or source promotion", len(seen), accepted, rejected)
	})
}
