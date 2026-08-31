package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/overgodb"
)

// TestAgentProvenance pins the per-step evidence walk: an approved
// mutation's provenance names the interaction, the transcript's tool
// call with the exact manual identity and result, the receipt chain
// from completed back to admitted, and the committed grant decision --
// while an inspection's provenance carries no receipts and no
// decision, because inspections are effect-free.
func TestAgentProvenance(t *testing.T) {
	handler := agentTestHandler(t, func(ctx context.Context, store *overgodb.Store) {
		manuals, err := agenttool.StandardManuals()
		if err != nil {
			t.Fatal(err)
		}
		write, err := agenttool.NewManual(agenttool.Manual{
			Name: "store.commit", Description: "A mutation manual for the provenance walk.",
			Effect:    agenttool.EffectMutation,
			Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeRepository, Value: "overgodb"}}},
			Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: "git", Args: []string{"status"}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := agenttool.PublishArgvPolicy(ctx, store, []string{"git"}); err != nil {
			t.Fatal(err)
		}
		if _, err := agenttool.PublishManualCatalog(ctx, store, append(manuals, write)); err != nil {
			t.Fatal(err)
		}
	})
	if code := serveTestRequest(handler, http.MethodPost, "/agent/step",
		`{"session":"walk","tool":"store.head"}`); code.Code != http.StatusOK {
		t.Fatalf("inspection status=%d body=%s", code.Code, code.Body.String())
	}
	preview := serveTestRequest(handler, http.MethodPost, "/agent/approval",
		`{"session":"walk","tool":"store.commit","arguments":{}}`)
	var previewed struct {
		Operation string `json:"operation"`
	}
	if err := json.Unmarshal(preview.Body.Bytes(), &previewed); err != nil || previewed.Operation == "" {
		t.Fatalf("preview status=%d operation=%q %v", preview.Code, previewed.Operation, err)
	}
	if code := serveTestRequest(handler, http.MethodPost, "/agent/step",
		`{"session":"walk","tool":"store.commit","arguments":{},"approval":"`+previewed.Operation+`"}`); code.Code != http.StatusOK {
		t.Fatalf("mutation status=%d body=%s", code.Code, code.Body.String())
	}

	mutation := serveTestRequest(handler, http.MethodGet, "/agent/provenance?session=walk&step=2", "")
	body := mutation.Body.String()
	if mutation.Code != http.StatusOK ||
		!strings.Contains(body, `"call_id":"walk-step-2"`) ||
		!strings.Contains(body, `"tool":"store.commit"`) ||
		!strings.Contains(body, `"manual"`) ||
		!strings.Contains(body, `"completed"`) ||
		!strings.Contains(body, `"running"`) ||
		!strings.Contains(body, `"admitted"`) ||
		!strings.Contains(body, `"answer":"grant"`) ||
		!strings.Contains(body, `"result"`) {
		t.Fatalf("mutation provenance status=%d body=%s", mutation.Code, body)
	}

	inspection := serveTestRequest(handler, http.MethodGet, "/agent/provenance?session=walk&step=1", "")
	inspectionBody := inspection.Body.String()
	if inspection.Code != http.StatusOK ||
		!strings.Contains(inspectionBody, `"tool":"store.head"`) ||
		strings.Contains(inspectionBody, `"receipts"`) ||
		strings.Contains(inspectionBody, `"decision"`) {
		t.Fatalf("inspection provenance status=%d body=%s", inspection.Code, inspectionBody)
	}

	absent := serveTestRequest(handler, http.MethodGet, "/agent/provenance?session=walk&step=9", "")
	if absent.Code != http.StatusNotFound {
		t.Fatalf("absent step status=%d body=%s", absent.Code, absent.Body.String())
	}
}
