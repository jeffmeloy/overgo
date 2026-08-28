package overgodb_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestWebhookDeliveryLedgerIdempotency(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := (artifact.DocumentContract{
		Kind: artifact.KindFile, MediaType: "application/octet-stream", Schema: "overgo/webhook-payload/v1",
	}).ContentBytes(bytes.Repeat([]byte("webhook-payload"), 512))
	if err != nil {
		t.Fatal(err)
	}
	records := make([]artifact.Content, 2)
	for index := range records {
		records[index], err = artifact.JSONContent(
			artifact.JSONContract(artifact.KindEvidence, "overgo/webhook-cas-fixture/v1"),
			struct {
				Contender int         `json:"contender"`
				Payload   artifact.ID `json:"payload"`
			}{Contender: index, Payload: payload.Descriptor.ID},
		)
		if err != nil {
			t.Fatal(err)
		}
	}
	const alias = "automation/webhook-delivery/cas-fixture"
	start := make(chan struct{})
	results := make(chan struct {
		index int
		err   error
	}, len(records))
	var wait sync.WaitGroup
	for index := range records {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			batch, batchErr := artifact.NewDocumentBatch(
				"webhook/cas/contender/"+records[index].Descriptor.ID.String(),
				[]artifact.Content{payload, records[index]},
				artifact.DependencyLineage(records[index].Descriptor.ID, payload.Descriptor.ID),
				[]artifact.AliasBinding{{Name: alias, Target: records[index].Descriptor.ID}},
			)
			if batchErr == nil {
				<-start
				_, batchErr = artifact.CommitBatch(ctx, store, batch)
			}
			results <- struct {
				index int
				err   error
			}{index: index, err: batchErr}
		}(index)
	}
	close(start)
	wait.Wait()
	close(results)
	winner := -1
	loser := -1
	for result := range results {
		switch {
		case result.err == nil:
			if winner >= 0 {
				t.Fatal("more than one blind alias contender committed")
			}
			winner = result.index
		case errors.Is(result.err, overgodb.ErrAliasConflict):
			loser = result.index
		default:
			t.Fatalf("contender %d error = %v", result.index, result.err)
		}
	}
	if winner < 0 || loser < 0 {
		t.Fatalf("CAS result winner=%d loser=%d", winner, loser)
	}
	bound, found, err := store.ResolveAlias(ctx, alias)
	if err != nil || !found || bound != records[winner].Descriptor.ID {
		t.Fatalf("alias = (%s, %v, %v)", bound, found, err)
	}
	if _, found, err := store.Artifact(ctx, records[loser].Descriptor.ID); err != nil || found {
		t.Fatalf("losing record published: found=%v err=%v", found, err)
	}
	assertBlobBytes(t, store, payload.Descriptor.ID, payload.Data)
	corrupt := bytes.Clone(payload.Data)
	corrupt[0] ^= 0xff
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "webhook/cas/corrupt", Contents: []artifact.Content{{Descriptor: payload.Descriptor, Data: corrupt}},
	}); err == nil {
		t.Fatal("conflicting payload bytes committed")
	}
	assertBlobBytes(t, store, payload.Descriptor.ID, payload.Data)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	bound, found, err = store.ResolveAlias(ctx, alias)
	if err != nil || !found || bound != records[winner].Descriptor.ID {
		t.Fatalf("reopened alias = (%s, %v, %v)", bound, found, err)
	}
	assertBlobBytes(t, store, payload.Descriptor.ID, payload.Data)
	parents, err := store.Parents(ctx, bound)
	if err != nil || len(parents) != 1 || parents[0].Parent != payload.Descriptor.ID {
		t.Fatalf("reopened lineage = (%+v, %v)", parents, err)
	}
}

func assertBlobBytes(t *testing.T, reader artifact.Reader, id artifact.ID, want []byte) {
	t.Helper()
	_, stream, found, err := reader.OpenContent(context.Background(), id)
	if err != nil || !found {
		t.Fatalf("blob = (%v, %v)", found, err)
	}
	got, err := io.ReadAll(stream)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("blob bytes = (%d, %v)", len(got), err)
	}
}
