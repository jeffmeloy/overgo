package main

import (
	"bytes"
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
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestMediaEvidenceBinding(t *testing.T) {
	testskip.Short(t)
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
	audit := func(ctx context.Context) error {
		refs := []artifact.ID{inputs.Bundle.Native, inputs.Bundle.Host, inputs.Bundle.Acquisition}
		refs = append(refs, inputs.Bundle.Processing...)
		refs = append(refs, inputs.Bundle.Failed...)
		refs = append(refs, slices.Collect(maps.Values(inputs.Bundle.Harnesses))...)
		refs = append(refs, slices.Collect(maps.Values(inputs.Bundle.Loaders))...)
		for _, source := range inputs.Bundle.SourceFiles {
			refs = append(refs, source.Evidence)
		}
		if err := artifact.ReadContents(ctx, reader, refs, func(artifact.Content) error { return nil }); err != nil {
			return err
		}
		protocol, err := read(ctx, inputs.Bundle.Protocol)
		if err != nil {
			return err
		}
		environment, err := read(ctx, inputs.Bundle.Environment)
		if err != nil {
			return err
		}
		if _, err := checkMediaCapacityBinding(inputs.Bundle, protocol, environment); err != nil {
			return err
		}
		for _, check := range inputs.Bundle.Checks {
			receipt, err := read(ctx, check.Evidence)
			if err != nil {
				return err
			}
			if err := checkMediaTestReceipt(check.Name, receipt, inputs.Assertions[check.Name]); err != nil {
				return err
			}
		}
		return nil
	}
	if err := audit(t.Context()); err != nil {
		t.Fatal(err)
	}
	t.Run("missing native fixture", func(t *testing.T) {
		reader = verificationContentFaultReader{Reader: store, id: inputs.Bundle.Native, absent: true}
		defer func() { reader = store }()
		if err := audit(t.Context()); err == nil {
			t.Fatal("missing native fixture gained audit credit")
		}
	})
	t.Logf("%d retained receipt contracts audited against input %s; historical scope retained", len(inputs.Bundle.Checks), inputID)
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
}
