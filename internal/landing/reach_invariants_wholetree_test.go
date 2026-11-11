//go:build wholetree

// The reach invariants run the lane discovery over the repository, so go test would count the tree among this
// package's inputs: built apart, they run in the landing's invariants stage, and the commit tool's other tests are
// reused until their own code changes (test-reuse).

package landing

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/testcheck"
)

// The lane sets reach rows name, each written once. Every lane a set names
// must be in the gate's or the cycle's table, so a renamed or removed lane
// fails the table rather than silently emptying the rows that name its set.
var (
	// servingLanes build or run what serves a model: the page at the gate,
	// and in the cycle the served-model journeys and the speech
	// acceptances.
	servingLanes = []string{"webui", "model-journey", "audio"}
	// browserOnlyLanes drive a browser and never the device. Every other
	// gate lane is a device lane, so a lane added to the gate's table counts
	// as one until it is named here.
	browserOnlyLanes = []string{"webui"}
)

// TestReachInvariants pins the boundaries between a changed path and the
// lanes it must never reach, and between a moved suite and the cycle lane
// that runs it, one row per invariant, all selected by one discovery over
// each real lane table, the way the lanes stage and validate -run select
// them. A row states what must never come back, a set of lanes the path
// stays out of, never the lanes it reaches today, so a lane it may
// legitimately reach later is no failure. A production row also holds every
// lane the path does reach to compiling it into that lane's production
// closure in the lane's build context, the code its tests build and run: a
// test import alone would make every such edit pay for a lane that never
// runs it, and the cure is moving the fixture that needs the import out of
// the lane or importing the smaller owner it uses. A row naming a cycle lane
// holds a suite moved off the gate: an edit to it must reach that lane, or
// the cycle would never run what the gate gave up. No gate lane builds with
// the cycle's tags, so a landing compiles none of the tests behind them.
//
// To pin a new boundary, add a row, never a test function: a representative
// path that exists, the named set it must not reach (add a set beside the
// others when none fits), whether only production code may carry it into a
// lane, the cycle lane that runs it when it is a moved suite, and the
// reason in one line.
func TestReachInvariants(t *testing.T) {
	t.Parallel()
	repo := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	everyLane := slices.Concat(lanes, cycleLanes)
	for _, name := range slices.Concat(servingLanes, browserOnlyLanes) {
		if !slices.ContainsFunc(everyLane, func(candidate lane) bool { return candidate.name == name }) {
			t.Fatalf("no lane is named %s; the rows naming its set would hold nothing", name)
		}
	}
	for _, candidate := range everyLane {
		if slices.Contains(strings.Split(candidate.tags, ","), pairTag) {
			t.Errorf("lane %s builds with %s; it would answer the paired comparisons whole each time it runs", candidate.name, pairTag)
		}
	}
	if !slices.Contains(strings.Split(checkedTags, ","), pairTag) {
		t.Errorf("vet reads with %s, without %s; a paired comparison would go unchecked", checkedTags, pairTag)
	}
	var gateLanes, deviceLanes []string
	for _, candidate := range lanes {
		if slices.ContainsFunc(strings.Split(candidate.tags, ","), func(tag string) bool { return slices.Contains(strings.Split(CycleTags, ","), tag) }) {
			t.Errorf("gate lane %s builds with the cycle's tags %s; a landing would run the suites the cycle owns", candidate.name, candidate.tags)
		}
		gateLanes = append(gateLanes, candidate.name)
		if !slices.Contains(browserOnlyLanes, candidate.name) {
			deviceLanes = append(deviceLanes, candidate.name)
		}
	}
	rows := []struct {
		path         string
		mustNotReach []string
		production   bool
		cycle        string
		reason       string
	}{
		{"internal/plan/plan.go", deviceLanes, true, "", "the plan is the loop's work list; editing it runs no model or kernel"},
		{"internal/landing/landing.go", deviceLanes, true, "", "the gate chooses the lanes; no lane runs the gate"},
		{"internal/loop/driver.go", deviceLanes, true, "", "the loop driver dispatches rows; no speech model or CUDA suite runs it"},
		{"internal/scorecard/scorecard.go", deviceLanes, true, "", "the scorecard reads recorded evidence; it measures nothing on the device"},
		{"internal/devicegrad/windowed_attention_backward_cuda_windows.go", servingLanes, false, "", "devicegrad holds kernels only trainers call, and no serving package links a trainer"},
		{"internal/devicegrad/windowed_attention_backward_cuda_windows_test.go", servingLanes, false, "", "a trainer kernel's test compiles only into devicegrad's own test binary"},
		{"internal/densecausaltrain/train_resident_cuda_windows.go", servingLanes, false, "", "the device dense-causal trainer is imported only by the commands that train"},
		{"internal/scratchtrain/resident_cuda_windows.go", servingLanes, false, "", "the device scratch trainer is imported only by trainers, never by the model builder the server links"},
		{"internal/e4btrain/model.go", servingLanes, false, "", "E4B training is a trainer; serving E4B never links it"},
		{"internal/devicemath/cuda_scope_windows.go", servingLanes, false, "", "no serving path calls devicemath; the served-model lanes link only host training"},
		{"internal/audioparity/vad_acceptance_test.go", gateLanes, false, "audio", "the speech acceptances serve the recorded models against the reference store"},
		{"cmd/server/main.go", gateLanes, false, "model-journey", "the served-model journeys build the server command and serve real models"},
		{"internal/server/webui_browser_firstrun_test.go", deviceLanes, false, "model-journey", "the first-run journey serves real models through the swap proxy"},
		{"internal/server/e4b_modalities_windows_test.go", deviceLanes, false, "models", "E4B over every HTTP route serves the real model"},
		{"internal/server/webui_library_projector_test.go", deviceLanes, false, "models", "the library projector hashes a stored model and projector pair"},
		{"internal/thoughtbank/load_model_test.go", gateLanes, false, "models", "the real-checkpoint suite opens Fractale beside the data"},
		{"internal/latentvideo/runtime_cuda_windows_test.go", gateLanes, false, "models", "Wan's runtime and reference edit open Wan2.1-T2V-1.3B beside the data"},
		{"internal/quant/quantize_weighted_windows_test.go", gateLanes, false, "models", "the quantizer's pinned check reads the reference build beside the data"},
		{"internal/diffusionimage/resident_sampling_cuda_windows_test.go", gateLanes, false, "models", "every device test of diffusionimage samples the real SimpleDiffusion checkpoint"},
		{"internal/mediacapability/video_cuda_windows_test.go", gateLanes, false, "models", "media capability activates real Wan, LiveEdit, Krea and RxBrain checkpoints"},
		{"internal/mediacapability/image_embedding_test.go", gateLanes, false, "models", "the image embedding capability loads the real DINO checkpoint"},
		{"internal/sensenovarecipe/runtime_cuda_windows_test.go", gateLanes, false, "models", "SenseNova generates from the real 8B checkpoint"},
		{"internal/sensenovarecipe/activate_model_test.go", gateLanes, false, "models", "SenseNova's activation derives from the real 8B checkpoint"},
		{"internal/adaptiveparity/qwen35_training_cuda_windows_test.go", gateLanes, false, "models", "adaptive parity trains real checkpoints beside the data"},
		{"internal/composition/composite_generation_parity_windows_test.go", gateLanes, false, "models", "composite generation runs the real Wan checkpoint"},
		{"internal/inference/runner_integration_test.go", gateLanes, false, "models", "the native oracles open real GGUFs"},
		{"internal/inference/qwen35_9b_serving_smoke_test.go", gateLanes, false, "models", "the 9B serving smoke loads the real Qwen3.5-9B"},
		{"internal/latentimage/krea_checkpoint_cuda_windows_model_test.go", gateLanes, false, "models", "the resident Krea checks open the real 12.8B checkpoint"},
		{"internal/latentimage/krea_checkpoint_model_test.go", gateLanes, false, "models", "the host Krea checks decode and encode with the real checkpoint"},
		{"internal/routedlm/flowterminal_model_test.go", gateLanes, false, "models", "the flow terminal runs the real SenseNova artifact"},
		{"internal/speechsynth/train_step_model_test.go", gateLanes, false, "models", "joint Muon training steps the real speech artifact"},
		{"internal/tabularicl/predict_wall_test.go", gateLanes, false, "models", "the predict wall loads every real TabICL head"},
		{"cmd/benchmark-report/accepted_coverage_test.go", gateLanes, false, "models", "MMLU-Pro coverage reads the live store's evaluations, so a store write would otherwise refuse an unrelated landing"},
		{"cmd/validate/verification_workflow_model_test.go", gateLanes, false, "models", "the media report projection reads the live store's activations, so an activation change would otherwise refuse an unrelated landing"},
		{"cmd/validate/accepted_image_video_model_test.go", gateLanes, false, "models", "the image-video coverage reads its accepted runs and outputs from the live store"},
		{"cmd/validate/image_video_baseline_model_test.go", gateLanes, false, "models", "the loader baseline reads its retained acquisitions from the live store"},
		{"cmd/validate/image_video_composition_model_test.go", gateLanes, false, "models", "the composition bundle reads its retained receipts from the live store"},
		{"cmd/validate/image_video_cost_model_test.go", gateLanes, false, "models", "the cost attribution reads the live store's active recipes and the real Wan metadata"},
		{"cmd/validate/image_video_latent_image_model_test.go", gateLanes, false, "models", "the latent image bundle reads its retained runs from the live store"},
		{"cmd/validate/image_video_memory_model_test.go", gateLanes, false, "models", "the memory comparison reads its retained acquisitions from the live store"},
		{"cmd/validate/image_video_processing_model_test.go", gateLanes, false, "models", "the processing comparison reads its retained acquisitions from the live store"},
		{"cmd/validate/image_video_protocol_model_test.go", gateLanes, false, "models", "the frozen protocol reads its cases' runs and outputs from the live store"},
		{"cmd/validate/image_video_routed_image_model_test.go", gateLanes, false, "models", "the routed image bundle reads its retained runs from the live store"},
		{"cmd/validate/image_video_shared_components_model_test.go", gateLanes, false, "models", "the shared component comparison reads its retained acquisitions from the live store"},
		{"cmd/validate/image_video_transfer_model_test.go", gateLanes, false, "models", "the transfer comparison reads its retained acquisitions from the live store"},
		{"cmd/validate/image_video_wan_model_test.go", gateLanes, false, "models", "the Wan GIF counterexamples decode the retained output from the live store"},
		{"cmd/validate/media_model_test.go", gateLanes, false, "models", "the retained measurement projection reads the live store's verifiers"},
		{"cmd/validate/models_report_model_test.go", gateLanes, false, "models", "the model report renders against the live store"},
		{"cmd/validate/musr_normalization_model_test.go", gateLanes, false, "models", "MuSR normalization replays retained scores from the live store"},
		{"cmd/validate/qwen4_ifeval_model_test.go", gateLanes, false, "models", "the Qwen 4B IFEval acquisition reads retained responses from the live store"},
		{"cmd/longform/accepted_baseline_model_test.go", gateLanes, false, "models", "the historical guard audit reads the live store's guard records"},
		{"internal/evaluation/bbh_native_model_test.go", gateLanes, false, "models", "the native BBH protocol reads its retained oracle from the live store"},
		{"internal/evaluation/ifeval_language_model_test.go", gateLanes, false, "models", "IFEval language parity reads its retained oracle from the live store"},
		{"internal/evaluation/ifeval_retained_text_model_test.go", gateLanes, false, "models", "IFEval native parity reads retained responses and scores from the live store"},
		{"internal/evaluation/ifeval_tokenizer_model_test.go", gateLanes, false, "models", "IFEval tokenizer parity reads its retained oracle from the live store"},
		{"internal/evaluation/qwen4_ifeval_scores_model_test.go", gateLanes, false, "models", "the Qwen 4B IFEval scores are read from the live store"},
		{"internal/evaluation/qwen9_ifeval_scores_model_test.go", gateLanes, false, "models", "the Qwen 9B IFEval scores are read from the live store"},
		{"internal/server/generation_store_workspace_model_test.go", gateLanes, false, "models", "the store media workspace lists the live store's activations and generates from a stored model"},
		{"internal/dataset/speech_source_acceptance_model_test.go", gateLanes, false, "models", "speech materialization reads the pinned LibriSpeech corpus beside the data"},
		{"internal/modelintake/projection_test.go", gateLanes, false, "models", "projection intake hashes the stored model and projector pair"},
		{"cmd/train/route_tabular_tabfm_test.go", gateLanes, false, "models", "the TabFM objective trains the real classification head"},
		{"cmd/validate/accepted_coverage_report_test.go", gateLanes, false, "models", "the coverage projection reads the live store's accepted evidence"},
		{"cmd/validate/accepted_modalities_model_test.go", gateLanes, false, "models", "the modality coverage reads the live store's accepted evidence"},
		{"cmd/evaluate/transcription_projected_windows_test.go", gateLanes, false, "models", "projected transcription serves the real E4B"},
		{"cmd/longform/e4b_recovery_windows_test.go", gateLanes, false, "models", "long-form recovery serves the real E4B at 32768 tokens"},
		{"cmd/recipe/projection_execution_windows_test.go", gateLanes, false, "models", "projection execution serves the real E4B"},
		{"internal/vitencoder/encoder_test.go", gateLanes, false, "models", "the DINO reference loads installed weights"},
		{"internal/seriesforecast/native_declaration_test.go", gateLanes, false, "models", "forecast declaration parity loads installed weights"},
		{"internal/rae/autoencoder_test.go", gateLanes, false, "models", "autoencoder parity loads the installed encoder and decoder"},
		{"internal/seq2seq/upstream_calls_model_test.go", gateLanes, false, "needle-reference", "the needle reference compares 400 cases without selecting training experiments"},
	}
	gateReaches := testcheck.Must(discover(repo, lanes...))(t)
	cycleReaches := testcheck.Must(discover(repo, cycleLanes...))(t)
	reaches := func(candidate lane, paths []string) bool {
		if slices.ContainsFunc(lanes, func(gate lane) bool { return gate.name == candidate.name }) {
			return gateReaches(candidate, paths).by != ""
		}
		return cycleReaches(candidate, paths).by != ""
	}
	productions := map[string]gosource.BuildSelection{}
	for _, row := range rows {
		if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(row.path))); err != nil {
			t.Errorf("row %s holds nothing; point it at a path that still carries the boundary: %v", row.path, err)
			continue
		}
		if moved := slices.IndexFunc(cycleLanes, func(candidate lane) bool { return candidate.name == row.cycle }); row.cycle != "" && (moved < 0 || cycleReaches(cycleLanes[moved], []string{row.path}).by == "") {
			t.Errorf("an edit to %s reaches no cycle lane %s, so the cycle never runs what the gate gave up: %s", row.path, row.cycle, row.reason)
		}
		for _, candidate := range everyLane {
			switch {
			case !reaches(candidate, []string{row.path}):
			case slices.Contains(row.mustNotReach, candidate.name):
				t.Errorf("an edit to %s reaches lane %s: %s", row.path, candidate.name, row.reason)
			case row.production:
				production, listed := productions[candidate.name]
				if !listed {
					production = testcheck.Must(gosource.HostBuildClosure(repo, append([]string{"-tags", candidate.tags}, candidate.packages...)...))(t)
					productions[candidate.name] = production
				}
				if !production.Files[row.path] {
					t.Errorf("an edit to %s reaches lane %s only through a test import; move the fixture that needs it out of the lane or import the smaller owner it uses", row.path, candidate.name)
				}
			}
		}
	}
}
