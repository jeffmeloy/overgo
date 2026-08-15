package artifact

import (
	"context"
	"testing"
)

type documentReader struct {
	Reader
	content Content
}

func (r documentReader) Content(context.Context, ID) (Content, bool, error) {
	return r.content, true, nil
}

type documentRepository struct {
	Repository
	alias   ID
	commits int
}

func (r *documentRepository) ResolveAlias(context.Context, string) (ID, bool, error) {
	return r.alias, true, nil
}

func (r *documentRepository) Commit(context.Context, Batch) (CommitID, error) {
	r.commits++
	return CommitID{1}, nil
}

func TestDocumentContractBuildsAndValidatesContent(t *testing.T) {
	contract := DocumentContract{
		Kind: KindDataset, MediaType: "application/test+json", Schema: "test/v1",
	}
	data := []byte(`{"version":1}`)
	content, err := contract.ContentBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	id := content.Descriptor.ID
	data[0] = 'x'
	if content.Data[0] != '{' {
		t.Fatal("content retained caller storage")
	}
	if err := contract.ValidateContent(content, id); err != nil {
		t.Fatal(err)
	}
	other := contract
	other.Schema = "test/v2"
	if err := other.ValidateContent(content, id); err == nil {
		t.Fatal("incompatible schema accepted")
	}
}

func TestDocumentContractRejectsInvalidIdentityAndBounds(t *testing.T) {
	contract := DocumentContract{Kind: KindRun, MediaType: "application/test+json", Schema: "test/v1"}
	data := []byte(`{"version":1}`)
	wrong, err := IdentifyBytes(KindRun, []byte("wrong"))
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateIdentity(wrong, data); err == nil {
		t.Fatal("wrong identity accepted")
	}
	if _, err := contract.Identify(nil); err == nil {
		t.Fatal("empty document accepted")
	}
	invalid := contract
	invalid.Kind = KindInvalid
	if _, err := invalid.Identify(data); err == nil {
		t.Fatal("invalid kind accepted")
	}
}

func TestCloneAliasBindingsOwnsCompareAndSetPointers(t *testing.T) {
	previous, err := IdentifyBytes(KindModel, []byte("previous"))
	if err != nil {
		t.Fatal(err)
	}
	target, err := IdentifyBytes(KindModel, []byte("target"))
	if err != nil {
		t.Fatal(err)
	}
	wantPrevious := previous
	source := []AliasBinding{{Name: "model", Target: target, Previous: &previous}}
	cloned := CloneAliasBindings(source)
	*source[0].Previous = target
	if cloned[0].Previous == source[0].Previous || *cloned[0].Previous != wantPrevious {
		t.Fatal("alias clone retained caller pointer")
	}
}

func TestDocumentBatchOwnsContent(t *testing.T) {
	contract := DocumentContract{
		Kind: KindDataset, MediaType: "application/test+json", Schema: "test/v1",
	}
	data := []byte(`{"version":1}`)
	id, err := contract.Identify(data)
	if err != nil {
		t.Fatal(err)
	}
	content, err := contract.Content(id, data)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := NewDocumentBatch("document/batch", []Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	content.Data[0] = 'x'
	if batch.Contents[0].Data[0] != '{' {
		t.Fatal("document batch retained caller bytes")
	}
}

func TestArtifactRepositoryGatesValidateInputs(t *testing.T) {
	contract := DocumentContract{
		Kind: KindDataset, MediaType: "application/test+json", Schema: "test/v1",
	}
	data := []byte(`{"version":1}`)
	id, err := contract.Identify(data)
	if err != nil {
		t.Fatal(err)
	}
	content, err := contract.Content(id, data)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := NewDocumentBatch("document/commit", []Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	repository := &documentRepository{alias: id}
	if _, err := CommitBatch(context.Background(), repository, batch); err != nil || repository.commits != 1 {
		t.Fatalf("commit gate = (%d, %v)", repository.commits, err)
	}
	resolved, ok, err := ResolveAlias(context.Background(), repository, "dataset")
	if err != nil || !ok || resolved != id {
		t.Fatalf("alias gate = (%s, %v, %v)", resolved, ok, err)
	}
	if _, err := CommitBatch(context.Background(), nil, batch); err == nil {
		t.Fatal("nil repository accepted")
	}
	invalid := batch
	invalid.Key = ""
	if _, err := CommitBatch(context.Background(), repository, invalid); err == nil || repository.commits != 1 {
		t.Fatal("invalid batch reached repository")
	}
}
