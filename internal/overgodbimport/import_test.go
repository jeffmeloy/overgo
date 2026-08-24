package overgodbimport

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

const importFixtureCommit = "0123456789abcdef0123456789abcdef01234567"

func TestLegacyProfileUpgradeAddsCurrentProvenance(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	legacy, err := json.Marshal(struct {
		Version      uint16                    `json:"version"`
		Architecture string                    `json:"architecture"`
		Policy       model.ArchitectureProfile `json:"policy"`
	}{Version: 1, Architecture: profile.Name, Policy: profile})
	if err != nil {
		t.Fatal(err)
	}
	upgraded, err := upgradeLegacyProfile(legacy)
	if err != nil {
		t.Fatal(err)
	}
	document, err := modelrecipe.ParseProfileDocument(upgraded)
	if err != nil {
		t.Fatal(err)
	}
	if document.Version != modelrecipe.ProfileVersion || len(document.Provenance) == 0 {
		t.Fatalf("upgraded profile = %+v", document)
	}
}

func TestImportResolvesFilesDocumentsManifestsAndLineage(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "weights.bin"), []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	records := []wireRecord{
		{Type: "artifact", Name: "weights", Kind: "tensor-set", Path: "weights.bin"},
		{Type: "manifest", Name: "model", Kind: "model", Components: []componentRecord{{
			Role: "weights", Name: "weights", Artifact: "weights",
		}}},
		{Type: "artifact", Name: "definition", Kind: "model-definition",
			Document:  json.RawMessage(`{"version":1,"model":{"$artifact":"model"}}`),
			MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "lineage", Child: "definition", Parent: "model", Relation: "depends-on"},
		{Type: "alias", Name: "models/active", Target: "definition"},
	}
	input := importStream(t, records, nil)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result, err := importRecords(context.Background(), store, root, nil, bytes.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Commit.Valid() || result.Source.Kind() != artifact.KindEvidence || len(result.Names) != 3 {
		t.Fatalf("import result = %+v", result)
	}
	resolved, ok, err := store.ResolveAlias(context.Background(), "models/active")
	if err != nil || !ok || resolved != result.Names["definition"] {
		t.Fatalf("resolved import = (%s, %t, %v)", resolved, ok, err)
	}
	parents, err := store.Parents(context.Background(), result.Names["definition"])
	if err != nil || len(parents) != 2 {
		t.Fatalf("definition parents = (%+v, %v)", parents, err)
	}
}

func TestImportRejectsCountDriftPathEscapeAndReferenceCycle(t *testing.T) {
	records := []wireRecord{{Type: "artifact", Name: "weights", Kind: "file", Path: "../weights.bin"}}
	input := importStream(t, records, nil)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := importRecords(context.Background(), store, t.TempDir(), nil, bytes.NewReader(input)); err == nil {
		t.Fatal("escaping path accepted")
	}
	cycle := []wireRecord{
		{Type: "artifact", Name: "a", Kind: "evidence", Document: json.RawMessage(`{"ref":{"$artifact":"b"}}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "b", Kind: "evidence", Document: json.RawMessage(`{"ref":{"$artifact":"a"}}`), MediaType: "application/test+json", Schema: "test/v1"},
	}
	if _, err := importRecords(context.Background(), store, t.TempDir(), nil, bytes.NewReader(importStream(t, cycle, nil))); err == nil {
		t.Fatal("reference cycle accepted")
	}
	counts := map[string]uint64{"kind:file": 2}
	if _, err := importRecords(context.Background(), store, t.TempDir(), nil, bytes.NewReader(importStream(t, records, counts))); err == nil {
		t.Fatal("count drift accepted")
	}
}

func TestImportRequiresRootAndUsesFullDigestIdentity(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "value.bin"), []byte("value"), 0o600); err != nil {
		t.Fatal(err)
	}
	input := importStream(t, []wireRecord{{
		Type: "artifact", Name: "value", Kind: "file", Path: "value.bin",
	}}, nil)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := importRecords(context.Background(), store, "", nil, bytes.NewReader(input)); err == nil {
		t.Fatal("empty artifact root accepted")
	}
	result, err := importRecords(context.Background(), store, root, nil, bytes.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	query, err := store.Query(context.Background(), overgodb.Query{
		MaxResults: 10, FromSequence: 1, ToSequence: 1,
		Projection: overgodb.ProjectCommits,
	})
	if err != nil || len(query.Commits) != 1 {
		t.Fatalf("import commit query = (%+v, %v)", query.Commits, err)
	}
	wantKey := "import/" + importFixtureCommit + "/" + exportDigest(input)
	if query.Commits[0].Key != wantKey || query.Commits[0].ID != result.Commit {
		t.Fatalf("import key = %q, want %q", query.Commits[0].Key, wantKey)
	}
}

func TestExternalEvidenceImportDerivesNativeRunEvaluationLineage(t *testing.T) {
	records := []wireRecord{
		{Type: "artifact", Name: "recipe", Kind: "recipe", Document: json.RawMessage(`{"name":"recipe"}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "environment", Kind: "evidence", Document: json.RawMessage(`{"name":"environment"}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "plan", Kind: "profile", Document: json.RawMessage(`{"name":"plan"}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "report", Kind: "evaluation", Document: json.RawMessage(`{"name":"report"}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "dataset", Kind: "dataset", Document: json.RawMessage(`{"name":"dataset"}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "run", Kind: "run", MediaType: runrecord.RunMediaType, Schema: runrecord.RunSchema, Document: json.RawMessage(`{"version":2,"recipe":{"$artifact":"recipe"},"outcome":"succeeded","inputs":[{"$artifact":"plan"}],"outputs":[{"$artifact":"report"}],"code_commit":"0123456789abcdef0123456789abcdef01234567","environment":{"$artifact":"environment"},"measured_ns":10,"phases":[{"phase":"validate","duration_ns":10}]}`)},
		{Type: "artifact", Name: "evaluation", Kind: "evaluation", MediaType: runrecord.EvaluationMediaType, Schema: runrecord.EvaluationSchema, Document: json.RawMessage(`{"version":1,"recipe":{"$artifact":"recipe"},"run":{"$artifact":"run"},"dataset":{"$artifact":"dataset"},"metrics":[{"name":"score","value":1,"direction":"maximize"}]}`)},
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result, err := importRecords(t.Context(), store, t.TempDir(), nil, bytes.NewReader(importStream(t, records, nil)))
	if err != nil {
		t.Fatal(err)
	}
	requireParent := func(child, parent artifact.ID, relation artifact.Relation) {
		t.Helper()
		edges, err := store.Parents(t.Context(), child)
		if err != nil {
			t.Fatal(err)
		}
		for _, edge := range edges {
			if edge.Parent == parent && edge.Relation == relation {
				return
			}
		}
		t.Fatalf("lineage %s -%s-> %s is absent: %+v", child, relation, parent, edges)
	}
	requireParent(result.Names["run"], result.Names["recipe"], artifact.RelationDependsOn)
	requireParent(result.Names["run"], result.Names["environment"], artifact.RelationDependsOn)
	requireParent(result.Names["run"], result.Names["plan"], artifact.RelationDependsOn)
	requireParent(result.Names["report"], result.Names["run"], artifact.RelationProducedBy)
	requireParent(result.Names["evaluation"], result.Names["recipe"], artifact.RelationDependsOn)
	requireParent(result.Names["evaluation"], result.Names["run"], artifact.RelationDependsOn)
	requireParent(result.Names["evaluation"], result.Names["dataset"], artifact.RelationDependsOn)
}

func importStream(t *testing.T, records []wireRecord, counts map[string]uint64) []byte {
	t.Helper()
	if counts == nil {
		counts = make(map[string]uint64)
		for _, record := range records {
			counts[recordCountKey(record)]++
		}
	}
	header := Header{Type: "manifest", Version: Version, SourceCommit: importFixtureCommit, Counts: counts}
	var output bytes.Buffer
	encode := func(value any) {
		content, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		output.Write(content)
		output.WriteByte('\n')
	}
	encode(header)
	for _, record := range records {
		encode(record)
	}
	return output.Bytes()
}
