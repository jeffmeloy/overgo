package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
)

// mmluProRequestsScript writes the fixture TestMMLUProAsksAsLMEvalDoes reads,
// from lm-eval's own task. It is kept where Python is kept, on the machine
// that ran it and not in git: the fixture is the data.
const mmluProRequestsScript = "scripts/python_golden_references/lm_eval_mmlu_pro_requests.py"

// recordingScorer keeps what it was asked and finds one candidate of each case likeliest.
//   - likeliest = the prompt's length modulo the candidates' count, so the cases of a fixture select apart
//   - a candidate's log-probability falls by one for each place it stands from the likeliest
type recordingScorer struct {
	prompts    *[]string
	candidates *[][]string
}

func (s recordingScorer) ScoreContinuations(_ context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	*s.prompts = append(*s.prompts, prompt)
	*s.candidates = append(*s.candidates, slices.Clone(candidates))
	scores := make([]sequencescore.Score, len(candidates))
	likeliest := len(prompt) % len(candidates)
	for index := range scores {
		scores[index] = sequencescore.Score{LogProbability: -float64(max(index-likeliest, likeliest-index) + 1), Tokens: 1}
	}
	return scores, nil
}

// TestMMLUProAsksAsLMEvalDoes holds what a likelihood scoring of MMLU-Pro
// gives a model to the task the suite is a port of, lm-eval's
// leaderboard_mmlu_pro, as lm-eval itself builds its requests. The task
// writes each choice as a bare letter and leaves the space before it to its
// target delimiter, so what is scored after "Answer:" is " A" and a
// demonstration ends "Answer: A". Scored glued, a letter is whatever token
// the tokenizer has for a colon and a letter together: one rare token for
// most letters and two for some, so that a small model's choice followed its
// tokenizer and not the question.
//
// For documents taken through the test split, the prompt and the candidates
// a likelihood scoring is handed must be lm-eval's context and continuations,
// with no demonstrations, as the store's suite is asked, and with the task's
// own five given as the grouped-choice kind gives a case its demonstrations.
func TestMMLUProAsksAsLMEvalDoes(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "fixtures", "lm_eval", "leaderboard_mmlu_pro_requests.json"))
	if err != nil {
		t.Fatalf("%v; write the fixture with %s", err, mmluProRequestsScript)
	}
	type document struct {
		Question string   `json:"question"`
		Options  []string `json:"options"`
		Answer   int      `json:"answer"`
		Category string   `json:"category"`
	}
	type requests []struct {
		Document      int      `json:"document"`
		Context       string   `json:"context"`
		Continuations []string `json:"continuations"`
	}
	var reference struct {
		TargetDelimiter string              `json:"target_delimiter"`
		Documents       map[string]document `json:"documents"`
		Demonstrations  []document          `json:"demonstrations"`
		Without         requests            `json:"without_demonstrations"`
		WithFive        requests            `json:"with_five_demonstrations"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Without) == 0 || len(reference.WithFive) != len(reference.Without) {
		t.Fatalf("the fixture holds %d and %d requests: %v", len(reference.Without), len(reference.WithFive), err)
	}
	if reference.TargetDelimiter != targetDelimiter {
		t.Fatalf("lm-eval's target delimiter is %q and overgo's %q", reference.TargetDelimiter, targetDelimiter)
	}
	demonstrations := make([]MultipleChoiceCase, len(reference.Demonstrations))
	for index, demonstration := range reference.Demonstrations {
		letters := make([]string, len(demonstration.Options))
		for at := range letters {
			letters[at] = targetDelimiter + choiceLetters[at]
		}
		demonstrations[index] = MultipleChoiceCase{
			Name: fmt.Sprintf("validation/%d", index), Prompt: letteredPrompt(demonstration.Question, demonstration.Options), Candidates: letters, Answer: demonstration.Answer,
		}
	}
	for shots, asked := range map[int]requests{0: reference.Without, len(demonstrations): reference.WithFive} {
		var cases []storeCase
		for _, request := range asked {
			source, found := reference.Documents[fmt.Sprint(request.Document)]
			if !found {
				t.Fatalf("the fixture asks about document %d and does not hold it", request.Document)
			}
			cases = append(cases, storeCase{entry: "mmlu-pro/" + source.Category + "/test", subset: source.Category, ordinal: request.Document, fields: rawFields(t, map[string]any{
				"question": source.Question, "options": source.Options, "answer_index": source.Answer, "category": source.Category,
			})})
		}
		assembled, _, err := assembleMMLUProSuite(cases)
		if err != nil {
			t.Fatal(err)
		}
		suite := assembled.(GroupedChoiceSuite)
		for index := range suite.Cases {
			suite.Cases[index].Demonstrations = demonstrations[:shots]
		}
		compiled, err := CompileGroupedChoice(suite)
		if err != nil {
			t.Fatal(err)
		}
		for index, scored := range compiled.choice.delimited().Cases {
			if scored.Prompt != asked[index].Context || !slices.Equal(scored.Candidates, asked[index].Continuations) {
				t.Errorf("with %d demonstrations, %s is asked as\n%q\nwith candidates %q, where lm-eval asks\n%q\nwith %q",
					shots, scored.Name, scored.Prompt, scored.Candidates, asked[index].Context, asked[index].Continuations)
			}
		}
	}
}

// mmluProProtocol is one protocol MMLU-Pro is asked under, with the identity digests trunk a91b29c42 bound for it through the MMLU-Pro kind.
//   - fixture = one digest for each drawn case of mmluProFixture alone, then one for the fixture whole
//   - store = one digest for each draw of the store's MMLU-Pro cases, and "each" = the digest of every case's digest alone in the catalog's order
type mmluProProtocol struct {
	authorities ExactAuthorities
	fixture     []string
	store       map[string]string
}

// mmluProStoreDatasets are the catalog datasets the store's MMLU-Pro cases were records of where the store digests were pinned.
const mmluProStoreDatasets = "[dataset:sha256:bf1b700749ef7d852dc83b816001fdca1bfae27adb3aa1c3c7d83c90ef5fdee1]"

func mmluProProtocols() map[string]mmluProProtocol {
	raw, chat, reasoned := ListingAuthorities(), ListingAuthorities(), ListingAuthorities()
	chat.Execution.Prompting = PromptingChatTemplate
	reasoned.Execution.Prompting, reasoned.Execution.Reasoning = PromptingReasoning, exploratoryReasoning()
	return map[string]mmluProProtocol{
		PromptingRawCompletion.Label(): {authorities: raw, fixture: []string{
			"a170988bdc4a6ac758ace32bfc0dfc75c7570a04f3d0fb34c9d582f97b1f6cff", "9b2b11b1c68da7ef667ca7bd30a1ced2ef7badf5c3a37134bde53c40e962ef77",
			"9a390de10cfa4e5ac741111e91c8c3a928d12cdb280b80da2efc49d2f805fea7", "29040af7c0522c8ec5a074a196f865a7a49a9aee4802907ef0370f289ad2a9c8",
			"b2064aa95f4eaaba85a04198d011c69f5718ab0f0046959642f64cf7b0821e00",
		}, store: map[string]string{
			"share": "f354e926c766c6b3f2331726d6f2e45860cef4e3b5fc7495af1d7749994252ab", "whole": "6e59d8c4a9c33af622ca9761b043075911cef6fbf17d6e13bdce3297d337b63d",
			"spaced": "0f1aa936d59fe076e09fed90c8e87c0ee9e63db4f0fd3734a93226b4d9914c12", "each": "263fc185c1c40b683e97ed3605a3efe3fb6750f7c7dc75eab9ae02a90aef68d4",
		}},
		PromptingChatTemplate.Label(): {authorities: chat, fixture: []string{
			"3d0e39f7cc165c995ed4be780d8250179b3102e2c783f76c8d31f63a90ad0838", "fb42ed15a54350b2ef8bf0430b2784a73063fb37f22382787783369120978f97",
			"5e6affe6fa179b42eda14963379ae4c3b6aa94a260ec651a72de8ceae22a4ad9", "aefee906ed5853fed751dbbe324cd3a3fba11535a2a914d4960f72c00530698e",
			"5ab75b3bfd94fb1cd7f6557553164ed4bd22eb101b8102c34a4a31ffc4d9d196",
		}, store: map[string]string{
			"share": "3bdd846d601bbfb04eaa6fee1715d2941ef90f4fc9056b6be2206f5acf3e3ec0", "whole": "cc5e63cc0bb71862623416c4d21ab6addf47c10747a547a8750614c6b950d74a",
			"spaced": "26ad017fed00bbe36c7be8cc41ed52ffb80ca7c968fa5fd8178cdd3a8879084c", "each": "7af809ea87114dc221016f16d06daa86e2c1da87f4380a466bbc8981947d9b29",
		}},
		PromptingReasoning.Label(): {authorities: reasoned, fixture: []string{
			"e53b127274c9935f2f4bf7968b3d17f27978f80c1b802ceb3ac36e7d92836923", "401999687fc7cb1661d6eaa836b0fc998536aa77d4cd28c946379680ce2c1c63",
			"b5866e62ece236502ae31cba57be5938ed5633979e9532ceffc9b64f09f0f756", "ee99fc1772e341a6552e86c3d8913f8d455e381ff97a1d3f114e1e5aaffcbfe8",
			"3f5186803389b9815a1e5304457c9d2d5c0135168d95f54c78b26fe794f6f980",
		}, store: map[string]string{
			"share": "b6d985f63d873d087b6fa8c56dd0e673b0bf113961206392da33a714e2fb50ea", "whole": "e6938ed6a1c716929c123dba29aa3afffd5430bb0a7678333ca6fb53035cdb52",
			"spaced": "aa5a30f88a16b77627b9141ea6a09e2a577af9c5f27aa2bc1bb9e3a87737f420", "each": "446efdad34fcfe5251d9907fa2797ec5479a6fdb8a0c171dd1288fe68772c76b",
		}},
	}
}

// mmluProFixture is store records of MMLU-Pro that vary every field a case is read from.
//   - questions and options with quotes, markup, escapes, a tab, line breaks, padding and characters outside ASCII
//   - two to ten options, answers first, inside and last, three categories one of which two cases share
//   - the last record is left out of the draw, so no suite asks it
func mmluProFixture(t *testing.T) []storeCase {
	t.Helper()
	record := func(category string, ordinal int, question string, options []string, answer int) storeCase {
		return storeCase{entry: "mmlu-pro/" + category + "/test", subset: category, ordinal: ordinal, fields: rawFields(t, map[string]any{
			"question": question, "category": category, "answer_index": answer, "options": options,
		})}
	}
	undrawn := record("law", 4, "Undrawn", []string{"a", "b"}, 0)
	undrawn.undrawn = true
	return []storeCase{
		record("business", 0, "Which of \"these\" is <tagged> & escaped?\nSecond line\tand a tab",
			[]string{"café", "naïve", "日本語", "√2 ≈ 1.414", "a\\b", "<b>bold</b>", "x & y", "  padded  ", "line\nbreak", "last"}, 9),
		record("math", 1, "One plus one?", []string{"three", "two"}, 1),
		record("math", 2, "Which is prime?", []string{" five ", "four", "six"}, 0),
		record("law", 3, "A tort is", []string{"a crime", "a contract", "a statute", "a civil wrong"}, 3),
		undrawn,
	}
}

// mmluProIdentities binds the MMLU-Pro suite of cases as the store's suites are bound and digests its identities.
//   - identities = dataset, split, case profile, scorer, plan, comparison and answers, in that order
//   - answers = what the plan's execution is asked, each candidate after the delimiter as the scorer is handed it
func mmluProIdentities(t *testing.T, cases []storeCase, authorities ExactAuthorities) (string, []artifact.ID) {
	t.Helper()
	assembled, _, err := assembleMMLUProSuite(cases)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileGroupedChoice(assembled.(GroupedChoiceSuite))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindGroupedChoice(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	document, err := json.Marshal(assembled)
	if err != nil {
		t.Fatal(err)
	}
	if derived, err := CompileSuite(document, authorities); err != nil || derived.Plan().Identity() != plan.Identity() {
		t.Fatalf("the suite compiled from its document binds another plan: %v", err)
	}
	comparison, err := plan.comparisonIdentity()
	if err != nil {
		t.Fatal(err)
	}
	scored := compiled.choice.delimited()
	asked := make([]askedChoice, len(scored.Cases))
	for index, testCase := range scored.Cases {
		asked[index] = askedChoice{Name: testCase.Name, Prompt: testCase.Prompt, Candidates: testCase.Candidates}
	}
	answers, err := plan.observed(asked)
	if err != nil {
		t.Fatal(err)
	}
	ids := []artifact.ID{plan.body.Dataset, plan.body.Split, plan.body.CaseProfile, plan.body.Scorer, plan.Identity(), comparison, answers}
	named := make([]string, len(ids))
	for index, id := range ids {
		named[index] = id.String()
	}
	digest := sha256.Sum256([]byte(strings.Join(named, "\n")))
	return hex.EncodeToString(digest[:]), ids
}

// TestMMLUProIsAGroupedChoiceWithItsIdentitiesUnchanged holds that MMLU-Pro asked as a grouped-choice suite with the target delimiter binds what the MMLU-Pro kind bound.
//   - every protocol it is asked under => each case alone and the cases together bind the pinned dataset, case profile, scorer, plan, comparison and answers
//   - the store's MMLU-Pro cases => the same for the part a measurement run scores, the whole, forty taken evenly, and each case alone
//   - the scorer is handed each letter after the delimiter, and the answers, observations and category accuracies are the bytes they were
//   - the letters glued to the cue => another plan under raw completion, the same plan under the chat template
//   - a delimiter other than the target delimiter => refused, since the scorer's method could not name it
func TestMMLUProIsAGroupedChoiceWithItsIdentitiesUnchanged(t *testing.T) {
	t.Parallel()
	fixture := mmluProFixture(t)
	protocols := mmluProProtocols()
	for label, protocol := range protocols {
		for index, want := range protocol.fixture {
			cases := fixture
			if index < len(protocol.fixture)-1 {
				cases = fixture[index : index+1]
			}
			if got, ids := mmluProIdentities(t, cases, protocol.authorities); got != want {
				t.Errorf("under %s the fixture's suite %d binds %v, digest %s, want digest %s as the MMLU-Pro kind bound", label, index, ids, got, want)
			}
		}
	}

	t.Run("store", func(t *testing.T) {
		t.Parallel()
		roots, err := dataroot.Resolve(testutil.RepoRoot(t))
		if err != nil {
			t.Skipf("no store to read MMLU-Pro cases from: %v", err)
		}
		reader, err := overgodb.OpenReadOnly(roots.Store)
		if err != nil {
			t.Skipf("the store at %s is not readable: %v", roots.Store, err)
		}
		defer reader.Close()
		families, datasets, err := storeFamilies(t.Context(), reader, "mmlu-pro")
		if err != nil || fmt.Sprint(datasets["mmlu-pro"]) != mmluProStoreDatasets {
			t.Skipf("the store's MMLU-Pro cases are records of %v, the digests were pinned on those of %s: %v", datasets["mmlu-pro"], mmluProStoreDatasets, err)
		}
		whole := families["mmlu-pro"]
		share, spaced := slices.Clone(whole), slices.Clone(whole)
		drawShare(share, suiteShares["mmlu-pro"])
		if err := drawSpaced(spaced, 40, true); err != nil {
			t.Fatal(err)
		}
		var sharePlan artifact.ID
		for label, protocol := range protocols {
			for name, cases := range map[string][]storeCase{"share": share, "whole": whole, "spaced": spaced} {
				got, ids := mmluProIdentities(t, cases, protocol.authorities)
				if got != protocol.store[name] {
					t.Errorf("under %s the store's %s draw binds %v, digest %s, want digest %s", label, name, ids, got, protocol.store[name])
				}
				if name == "share" && label == PromptingRawCompletion.Label() {
					sharePlan = ids[4]
				}
			}
			each := sha256.New()
			for index := range whole {
				digest, _ := mmluProIdentities(t, whole[index:index+1], protocol.authorities)
				each.Write([]byte(digest + "\n"))
			}
			if got := hex.EncodeToString(each.Sum(nil)); got != protocol.store["each"] {
				t.Errorf("under %s the store's %d cases alone digest to %s, want %s", label, len(whole), got, protocol.store["each"])
			}
		}
		suites, _, err := DeriveStoreSuiteFamily(t.Context(), reader, protocols[PromptingRawCompletion.Label()].authorities, "mmlu-pro")
		if err != nil || len(suites) != 1 || suites[0].Plan().Identity() != sharePlan || suites[0].Descriptor().Kind != GroupedChoiceKind {
			t.Fatalf("the derived MMLU-Pro family is %d suites, the first binding another plan than %s or of another kind: %v", len(suites), sharePlan, err)
		}
	})

	raw := protocols[PromptingRawCompletion.Label()].authorities
	assembled, _, err := assembleMMLUProSuite(fixture)
	if err != nil {
		t.Fatal(err)
	}
	suite := assembled.(GroupedChoiceSuite)
	compiled, err := CompileGroupedChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	if got := compiled.choice.suite.Cases[1].Candidates; !slices.Equal(got, []string{"A", "B"}) {
		t.Fatalf("the suite's own candidates are %q, want the bare letters", got)
	}
	glued := suite
	glued.Delimiter = ""
	gluedPlan, err := CompileGroupedChoice(glued)
	if err != nil {
		t.Fatal(err)
	}
	for _, asked := range []prompting{PromptingRawCompletion, PromptingChatTemplate} {
		authorities := raw
		authorities.Execution.Prompting = asked
		with, withErr := BindGroupedChoice(compiled, authorities)
		without, withoutErr := BindGroupedChoice(gluedPlan, authorities)
		if withErr != nil || withoutErr != nil || (with.identity == without.identity) != (asked == PromptingChatTemplate) {
			t.Errorf("under %s the plan with the delimiter and the plan without it are the same plan: %v (%v, %v)", asked.Label(), with.identity == without.identity, withErr, withoutErr)
		}
	}
	glued.Delimiter = "\t"
	if _, err := CompileGroupedChoice(glued); err == nil {
		t.Error("a suite scored after a tab was compiled, and its scorer's method names the target delimiter")
	}

	plan, err := BindGroupedChoice(compiled, raw)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	var prompts []string
	var candidates [][]string
	report, err := EvaluateGroupedChoice(t.Context(), store, recordingScorer{prompts: &prompts, candidates: &candidates}, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompts) != len(suite.Cases) || prompts[1] != "One plus one?\nA. three\nB. two\nAnswer:" || !slices.Equal(candidates[1], []string{" A", " B"}) {
		t.Fatalf("the scorer was asked %q with %q", prompts, candidates)
	}
	observations, err := json.Marshal(report.Observations)
	if err != nil {
		t.Fatal(err)
	}
	groups, err := json.Marshal(report.Groups)
	if err != nil {
		t.Fatal(err)
	}
	const (
		pinnedAnswers      = "profile:sha256:b18cd03429d6ba014ebe2b851a79ba33e31506d0e9f834423a03067228fa91c0"
		pinnedObservations = "0b4637e4f62fe693ee3ac40d2dd949a992a761b15365c8d7606adec8b704ae98"
		pinnedGroups       = "c985254ee4246f185a648e3f3486c46599d2f66cb645857886370d95d8802bba"
	)
	if digest := sha256.Sum256(observations); report.Answers.String() != pinnedAnswers || hex.EncodeToString(digest[:]) != pinnedObservations {
		t.Errorf("the fixture's answers are %s and its observations %s, want %s and the bytes the MMLU-Pro kind wrote", report.Answers, observations, pinnedAnswers)
	}
	if digest := sha256.Sum256(groups); report.Accuracy != 0.5 || hex.EncodeToString(digest[:]) != pinnedGroups {
		t.Errorf("the fixture reads accuracy %v by category %s, want 0.5 and the bytes the MMLU-Pro kind wrote", report.Accuracy, groups)
	}
}
