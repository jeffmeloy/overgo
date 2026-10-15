package evaluation

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

type mediaInputRuntime struct {
	observedRuntime
	input    MediaInput
	producer artifact.ID
	resolves int
}

func (r *mediaInputRuntime) ResolveMedia(context.Context, MediaKind, string) (MediaInput, error) {
	r.resolves++
	return r.input, nil
}

func (r *mediaInputRuntime) Judge() string {
	id, _ := artifact.JSONID(artifact.KindProfile, r.input.Runtime)
	return id.String()
}

func (r *mediaInputRuntime) ProducingRun() artifact.ID { return r.producer }

func TestMediaEvidenceNamesConsumedInputsAndKeepsItsMethod(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture",
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := &mediaInputRuntime{producer: planID(t, artifact.KindRun, "answer producer")}
	campaign, err := NewCampaign(store, runtime, modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "candidate"),
		Definition: planID(t, artifact.KindModelDefinition, "candidate definition"),
		Recipe:     planID(t, artifact.KindRecipe, "candidate recipe"),
	}, environment, planTestCommit)
	if err != nil {
		t.Fatal(err)
	}
	base := MediaInput{
		Content:  planID(t, artifact.KindOutput, "pixels"),
		Runtime:  MediaRuntime{Model: planID(t, artifact.KindModel, "judge"), Recipe: planID(t, artifact.KindRecipe, "judge recipe")},
		Decode:   MediaDecode{Format: "image"},
		Request:  planID(t, artifact.KindFile, "full request"),
		Producer: planID(t, artifact.KindRun, "draw producer"),
	}
	suite := func(name, question, answer string) GeneratedAnswerSuite {
		return GeneratedAnswerSuite{Kind: GeneratedAnswerKind, Schema: "fixture", Source: "fixture", Judge: "descriptive label",
			Cases: []GeneratedAnswerCase{
				{Name: "one", Prompt: mediaPrompt(MediaImage, name, question), MaxTokens: exactMaxTokens, Answers: []string{answer}},
				{Name: "two", Prompt: mediaPrompt(MediaImage, name, "second question"), MaxTokens: exactMaxTokens, Answers: []string{"ok"}},
			}}
	}
	compile := func(input MediaInput, value GeneratedAnswerSuite) (CompiledSuite, artifact.ID, artifact.ID) {
		t.Helper()
		runtime.input, runtime.resolves = input, 0
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := campaign.CompileSuite(t.Context(), data)
		if err != nil {
			t.Fatal(err)
		}
		if runtime.resolves != 1 {
			t.Fatalf("one shared input resolved %d times", runtime.resolves)
		}
		var bound GeneratedAnswerSuite
		for _, content := range compiled.plan.authorities {
			if content.Descriptor.ID == compiled.plan.body.CaseProfile {
				if err := json.Unmarshal(content.Data, &bound); err != nil {
					t.Fatal(err)
				}
			}
		}
		asked := make([]string, len(bound.Cases))
		for i, testCase := range bound.Cases {
			asked[i] = testCase.Prompt
		}
		observed, err := compiled.plan.observed(asked)
		if err != nil {
			t.Fatal(err)
		}
		method, err := compiled.plan.comparisonIdentity()
		if err != nil {
			t.Fatal(err)
		}
		return compiled, method, observed
	}
	for _, generated := range []bool{false, true} {
		input := base
		if !generated {
			input.Content = planID(t, artifact.KindFile, "pixels")
			input.Request, input.Producer = artifact.ID{}, artifact.ID{}
		}
		original, method, observed := compile(input, suite("first.png", "question", "ok"))
		for _, change := range []struct {
			name                             string
			edit                             func(*MediaInput, *GeneratedAnswerSuite)
			samePlan, sameMethod, sameAnswer bool
		}{
			{"relocated bytes", func(_ *MediaInput, s *GeneratedAnswerSuite) { *s = suite("elsewhere.png", "question", "ok") }, true, true, true},
			{"changed bytes", func(m *MediaInput, _ *GeneratedAnswerSuite) {
				m.Content = planID(t, m.Content.Kind(), "different pixels")
			}, false, generated, false},
			{"changed runtime", func(m *MediaInput, _ *GeneratedAnswerSuite) {
				m.Runtime.Recipe = planID(t, artifact.KindRecipe, "other judge")
			}, false, !generated, false},
			{"changed projector", func(m *MediaInput, _ *GeneratedAnswerSuite) {
				m.Runtime.Projection = planID(t, artifact.KindRecipe, "other projector")
			}, false, !generated, false},
			{"changed question", func(_ *MediaInput, s *GeneratedAnswerSuite) { *s = suite("first.png", "other question", "ok") }, false, false, false},
			{"changed scoring", func(_ *MediaInput, s *GeneratedAnswerSuite) { s.Cases[0].Answers = []string{"no"} }, false, false, true},
			{"changed producer", func(m *MediaInput, _ *GeneratedAnswerSuite) {
				if generated {
					m.Producer = planID(t, artifact.KindRun, "another draw")
				}
			}, !generated, true, true},
			{"changed request", func(m *MediaInput, _ *GeneratedAnswerSuite) {
				if generated {
					m.Request = planID(t, artifact.KindFile, "another request")
				}
			}, !generated, !generated, true},
		} {
			t.Run(change.name+map[bool]string{true: "/generated", false: "/fixed"}[generated], func(t *testing.T) {
				changed, document := input, suite("first.png", "question", "ok")
				change.edit(&changed, &document)
				plan, nextMethod, nextAnswer := compile(changed, document)
				if (plan.plan.Identity() == original.plan.Identity()) != change.samePlan ||
					(nextMethod == method) != change.sameMethod || (nextAnswer == observed) != change.sameAnswer {
					t.Fatalf("same plan/method/answer = %t/%t/%t", plan.plan.Identity() == original.plan.Identity(), nextMethod == method, nextAnswer == observed)
				}
			})
		}
	}
	video := suite("clip", "question", "ok")
	for i := range video.Cases {
		_, name, question, _ := ParseMediaPrompt(video.Cases[i].Prompt)
		video.Cases[i].Prompt = mediaPrompt(MediaVideo, name, question)
	}
	input := base
	input.Decode = MediaDecode{Format: "gif", FPS: 1, Frames: 1}
	_, method, answer := compile(input, video)
	for _, decode := range []MediaDecode{
		{Format: "gif", FPS: 2, Frames: 1},
		{Format: "gif", FPS: 1, Frames: 2},
		{Format: "ffmpeg", FPS: 1, Frames: 1, Tool: planID(t, artifact.KindFile, "decoder")},
	} {
		input.Decode = decode
		_, changedMethod, changedAnswer := compile(input, video)
		if changedMethod == method || changedAnswer == answer {
			t.Fatal("changed decode kept media method or answer identity")
		}
	}
	// A valid old record remains readable but cannot qualify media evidence.
	value := suite("first.png", "question", "ok")
	current, _, _ := compile(base, value)
	publishPlanFixtureAuthorities(t, store, current.plan)
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "media fixture producers", Artifacts: []artifact.Descriptor{{ID: campaign.identity.Model}, {ID: base.Producer}, {ID: runtime.producer}}}); err != nil {
		t.Fatal(err)
	}
	result, err := campaign.Evaluate(t.Context(), current)
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.RequireEvaluation(t.Context(), store, result.Evaluation)
	if err != nil || !recordExecution(t.Context(), store, record).ExecutionKnown {
		t.Fatalf("bound media is unknown: %v", err)
	}
	parents, err := store.Parents(t.Context(), result.Report)
	if err != nil || !slices.ContainsFunc(parents, func(edge artifact.Lineage) bool { return edge.Parent == runtime.producer }) {
		t.Fatalf("answer producer lineage absent: %v", err)
	}
	report, found, err := artifact.ReadContent(t.Context(), store, result.Report)
	var observed generatedAnswerReport
	if err != nil || !found || json.Unmarshal(report.Data, &observed) != nil || observed.Observations[0].Producer != runtime.producer {
		t.Fatalf("original answer producer absent: %v", err)
	}
	value.Judge = runtime.Judge()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := CompileSuite(data, campaign.Authorities())
	if err != nil {
		t.Fatal(err)
	}
	calls := runtime.generateCalls
	if _, err := campaign.Evaluate(t.Context(), legacy); err == nil || runtime.generateCalls != calls {
		t.Fatal("unbound media executed")
	}
	oldResult, err := executeSuite(t.Context(), store, runtime, legacy)
	if err != nil {
		t.Fatal(err)
	}
	retained, err := campaign.finishEvaluation(t.Context(), legacy, oldResult, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	oldRecord, err := runrecord.RequireEvaluation(t.Context(), store, retained.Evaluation)
	if err != nil || recordExecution(t.Context(), store, oldRecord).ExecutionKnown {
		t.Fatalf("legacy media qualified: %v", err)
	}
	for _, incomplete := range []MediaInput{
		{Content: base.Content, Runtime: base.Runtime, Decode: base.Decode, Request: base.Request},
		{Content: base.Content, Runtime: base.Runtime, Decode: MediaDecode{Format: "ffmpeg", FPS: 1, Frames: 1}, Request: base.Request, Producer: base.Producer},
	} {
		runtime.input = incomplete
		if _, err := campaign.CompileSuite(t.Context(), data); err == nil {
			t.Fatal("incomplete media authority qualified")
		}
	}
}
