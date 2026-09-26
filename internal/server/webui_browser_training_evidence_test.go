package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// trainingEvidenceObservation is one DPO step as the trainer's trace records it.
type trainingEvidenceObservation struct {
	Step            int     `json:"step"`
	Loss            float64 `json:"loss"`
	PolicyChosen    float64 `json:"policy_chosen"`
	PolicyRejected  float64 `json:"policy_rejected"`
	PolicyMargin    float64 `json:"policy_margin"`
	ReferenceChosen float64 `json:"reference_chosen"`
	ReferenceReject float64 `json:"reference_rejected"`
	ReferenceMargin float64 `json:"reference_margin"`
	RelativeMargin  float64 `json:"relative_margin"`
	ChosenTokens    int     `json:"chosen_tokens"`
	RejectedTokens  int     `json:"rejected_tokens"`
	GradientL2      float64 `json:"gradient_l2"`
	UpdateL2        float64 `json:"update_l2"`
	LearningRate    float64 `json:"learning_rate"`
}

// TestWebUIBrowserTrainingEvidence drives the training page over two stored
// DPO runs: a run's trace draws its loss and margins as the measured values
// (a negative margin kept, nothing smoothed), the pair inspector steps
// through the chosen and rejected scores, and a run pinned as the baseline
// compares its last step with another run's in the checkpoint comparison.
func TestWebUIBrowserTrainingEvidence(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the training evidence leg uses Chromium through cmd/webui-lane")
	}
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "training evidence recipe")
	// publish stores one run with its checkpoint and the DPO trace it produced.
	publish := func(label string, observations []trainingEvidenceObservation) {
		t.Helper()
		checkpoint := testutil.ArtifactID(t, artifact.KindCheckpoint, label+" checkpoint")
		if _, err := store.Commit(ctx, artifact.Batch{Key: "training-evidence/" + label + "/facts", Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: checkpoint}}}); err != nil {
			t.Fatal(err)
		}
		run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, nil, []artifact.ID{checkpoint}, "")
		if err != nil {
			t.Fatal(err)
		}
		batch, err := run.Batch("training-evidence/" + label + "/run")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		trace, err := json.Marshal(struct {
			Run       artifact.ID                   `json:"run"`
			Objective string                        `json:"objective"`
			DPO       []trainingEvidenceObservation `json:"dpo"`
		}{run.ID, "dpo", observations})
		if err != nil {
			t.Fatal(err)
		}
		traceID, err := artifact.IdentifyBytes(artifact.KindEvidence, trace)
		if err != nil {
			t.Fatal(err)
		}
		descriptor := artifact.Descriptor{ID: traceID, Size: uint64(len(trace)), MediaType: "application/json"}
		if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
			Key: "training-evidence/" + label + "/trace", Contents: []artifact.Content{{Descriptor: descriptor, Data: trace}},
			Lineage: []artifact.Lineage{{Child: traceID, Parent: run.ID, Relation: artifact.RelationProducedBy}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	step := func(step int, loss, policy, reference float64) trainingEvidenceObservation {
		return trainingEvidenceObservation{
			Step: step, Loss: loss, PolicyChosen: policy, PolicyRejected: 0, PolicyMargin: policy,
			ReferenceChosen: reference, ReferenceReject: 0, ReferenceMargin: reference, RelativeMargin: policy - reference,
			ChosenTokens: 4 + step, RejectedTokens: 3, GradientL2: 0.5, UpdateL2: 0.25, LearningRate: 0.001,
		}
	}
	publish("baseline", []trainingEvidenceObservation{step(1, 0.7, -0.2, 0.1), step(2, 0.6, 0.3, 0.1)})
	publish("current", []trainingEvidenceObservation{step(1, 0.5, 0.4, 0.1), step(2, 0.4, 0.9, 0.1)})

	handler := newTestHandlerForRepository(t, store, &fakeGenerator{})
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { location.hash = 'runs'; return true; })()`)
	settle(`!!document.querySelector('#panel-runs.active')`)
	check(`(() => {
  window.runsPanel = () => document.querySelector('#panel-runs');
  window.runButtons = () => [...runsPanel().querySelectorAll('table button.link-button')];
  window.seriesValues = (title) => {
    const block = [...runsPanel().querySelectorAll('.evidence-block')].find(section => section.querySelector('.section-title')?.textContent === title);
    return block ? [...block.querySelectorAll('circle')].map(dot => dot.getAttribute('data-value')) : [];
  };
  window.rowsOf = (first) => { const table = [...runsPanel().querySelectorAll('table')].find(table => table.querySelector('th')?.textContent === first);
    return table ? [...table.rows].slice(1).map(row => [...row.cells].map(cell => cell.textContent)) : []; };
  return true;
})()`)
	settle(`runButtons().length === 2`)
	// The newest run lists first: the current run, then the baseline.
	check(`(() => { runButtons()[1].click(); return true; })()`)
	settle(`seriesValues('DPO loss').join(',') === '0.7,0.6'`)
	check(`seriesValues('Margin decomposition').includes('-0.2')`)
	// The pair inspector steps through the recorded scores.
	check(`rowsOf('scorer').find(row => row[0] === 'policy')[3] === '-0.2'`)
	check(`(() => { const select = runsPanel().querySelector('select[aria-label="observation step"]'); select.value = '1'; select.dispatchEvent(new Event('change')); return true; })()`)
	settle(`rowsOf('scorer').find(row => row[0] === 'policy')[3] === '0.3'`)
	// Pinned as the baseline, it compares with the other run's last step.
	check(`(() => { [...runsPanel().querySelectorAll('button')].find(button => button.textContent === 'Set comparison baseline').click(); runButtons()[0].click(); return true; })()`)
	settle(`rowsOf('measurement').some(row => row[0] === 'loss' && row[1] === '0.6' && row[2] === '0.4')`)
	check(`rowsOf('measurement').some(row => row[0] === 'policy_margin' && row[1] === '0.3' && row[2] === '0.9')`)
	webuilane.Leg(t, "training evidence leg", "two stored DPO runs drew their measured loss and margins, the pair inspector stepped through the recorded scores, and a pinned baseline compared with the other run's last step")
}
