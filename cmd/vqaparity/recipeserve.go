package main

// Recipe-gated serve: the RxBrain VQA pipeline driven THROUGH the activated
// recipe rather than the golden parity harness. It (1) resolves the active VQA
// recipe from the RepoDB store (asserting the model artifact, recipe definition,
// and evidence tier are published), (2) runs the real processor on an image +
// question (no golden ids), and (3) drives the shared device pipeline
// (runFullPipeline) with a decode-until-EOS budget. The generated answer is
// asserted exact and repeated for determinism.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/dataroot"
	"overgo/internal/hfbpe"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/patchtower"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/routedlm"
)

// decodeChain: token ids -> text via the checkpoint tokenizer.
func decodeChain(modelDir string, ids []int) (string, error) {
	tok, err := hfbpe.Load(modelDir)
	if err != nil {
		return "", err
	}
	return tok.Decode(ids), nil
}

// readEOSTokenIDs: generation_config.json eos_token_id (scalar or list) — the
// authoritative stop tokens for generation.
func readEOSTokenIDs(modelDir string) ([]int, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		return nil, err
	}
	var parsed struct {
		EOS json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("generation_config: %w", err)
	}
	if len(parsed.EOS) == 0 {
		return nil, fmt.Errorf("generation_config: no eos_token_id")
	}
	var list []int
	if err := json.Unmarshal(parsed.EOS, &list); err == nil {
		return list, nil
	}
	var scalar int
	if err := json.Unmarshal(parsed.EOS, &scalar); err != nil {
		return nil, fmt.Errorf("generation_config: eos_token_id shape: %w", err)
	}
	return []int{scalar}, nil
}

// resolveActiveVQA: gate — the serve runs only when an active VQA recipe is
// published for THIS artifact. Returns the model id + recipe (definition) id +
// tier for the round-trip log.
func resolveActiveVQA(ctx context.Context, repo, modelDir string) (modelID, recipeID, tier string, err error) {
	repository := repo
	if repository == "" {
		working, e := os.Getwd()
		if e != nil {
			return "", "", "", e
		}
		roots, e := dataroot.Resolve(working)
		if e != nil {
			return "", "", "", e
		}
		repository = roots.Store
	}
	hf, err := hfrepo.Open(modelDir)
	if err != nil {
		return "", "", "", err
	}
	inventory, err := modelartifact.FromHFRepository(hf)
	_ = hf.Close()
	if err != nil {
		return "", "", "", err
	}
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return "", "", "", err
	}
	defer store.Close()
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, inventory.Manifest.ID, recipe.TaskVQA)
	if err != nil {
		return "", "", "", err
	}
	if !active {
		return "", "", "", fmt.Errorf("no active vqa recipe for %s (activate first)", inventory.Manifest.ID)
	}
	return inventory.Manifest.ID.String(), activation.Definition.ID.String(), string(activation.Tier), nil
}

// runRecipeServe: serve the canonical case through the activated recipe.
func runRecipeServe(l *ladder, repo, imagePath, question string) error {
	ctx := context.Background()

	// ---- GATE: active VQA recipe must be published for this artifact -------
	modelID, recipeID, tier, err := resolveActiveVQA(ctx, repo, l.modelDir)
	if err != nil {
		return err
	}
	l.log(fmt.Sprintf("RECIPE serve GATE active vqa recipe resolved model=%s recipe=%s tier=%s", modelID, recipeID, tier))

	// ---- processor: real image + question -> input ids (NO golden) --------
	pre, err := patchtower.LoadPreprocessConfig(l.modelDir)
	if err != nil {
		return err
	}
	tok, err := hfbpe.Load(l.modelDir)
	if err != nil {
		return err
	}
	specials, err := routedlm.LoadPromptSpecials(l.modelDir, promptRoles)
	if err != nil {
		return err
	}
	if _, statErr := os.Stat(imagePath); statErr != nil {
		return fmt.Errorf("serve image not found: %s", imagePath)
	}
	rawImg, err := os.ReadFile(imagePath)
	if err != nil {
		return err
	}
	rgb, h, w, err := patchtower.DecodeImageBytesRGB(rawImg)
	if err != nil {
		return err
	}
	pixelValues, gridT, gridH, gridW, err := patchtower.PreprocessImage(pre, rgb, h, w)
	if err != nil {
		return err
	}
	inputIDs, err := routedlm.RenderVisionQAPrompt(tok, specials, question, gridH, gridW, pre.MergeSize)
	if err != nil {
		return err
	}
	positions := routedlm.ImageMaskPositions(inputIDs, specials)
	l.log(fmt.Sprintf("RECIPE serve PROCESSOR image=%s grid=[%d,%d,%d] promptLen=%d imageTokens=%d question=%q",
		filepath.Base(imagePath), gridT, gridH, gridW, len(inputIDs), len(positions), question))

	eosIDs, err := readEOSTokenIDs(l.modelDir)
	if err != nil {
		return err
	}

	// The 12-step parity golden is a TRUNCATION of the model's natural answer;
	// the real serve decodes until EOS. Two exactness anchors: (1) the generated
	// chain reproduces the proven golden chain as an exact prefix (bit-exact vs
	// the device-full harness), and (2) the decoded text begins with the golden
	// answer phrase and completes coherently, stopping at EOS.
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}
	goldenChain := dg.GeneratedTokens
	const wantPrefix = "The stovetop holds a metal pot on the left burner"
	const maxSteps = 64

	serveOnce := func(tag string) ([]int, string, time.Duration, fullResult, error) {
		pc, err := newPrefillContext(l.modelDir, inputIDs, positions, gridT, gridH, gridW)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		defer pc.src.Close()
		worker, err := device.New(0)
		if err != nil {
			return nil, "", 0, fullResult{}, fmt.Errorf("recipe serve worker: %w", err)
		}
		defer worker.Close()
		exe, err := executor.NewWithWorker(worker)
		if err != nil {
			return nil, "", 0, fullResult{}, fmt.Errorf("recipe serve executor: %w", err)
		}
		defer exe.Close()
		res, err := runFullPipeline(l, ctx, worker, exe, pc, pixelValues, fullOpts{maxSteps: maxSteps, eosIDs: eosIDs})
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		text, err := decodeChain(l.modelDir, res.generated)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		l.log(fmt.Sprintf("RECIPE serve %s chain=%v", tag, res.generated))
		l.log(fmt.Sprintf("RECIPE serve %s TEXT %q", tag, text))
		l.log(fmt.Sprintf("RECIPE serve %s MEASURE e2e=%s decode=%s (%d steps, %.3f ms/token) peak gpu.used=%dMiB",
			tag, res.e2eWall.Round(time.Millisecond), res.decodeWall.Round(time.Millisecond), res.steps,
			float64(res.decodeWall.Microseconds())/1000.0/float64(max1(res.steps)), res.peakMiB))
		return res.generated, text, res.e2eWall, res, nil
	}

	chain1, text1, e2e1, _, err := serveOnce("RUN1")
	if err != nil {
		return err
	}
	if len(chain1) < len(goldenChain) {
		return fmt.Errorf("recipe serve chain len %d < golden prefix %d", len(chain1), len(goldenChain))
	}
	if err := requireIntSliceEqual("recipe serve golden-prefix chain", chain1[:len(goldenChain)], goldenChain); err != nil {
		return err
	}
	if !strings.HasPrefix(text1, wantPrefix) {
		return fmt.Errorf("recipe serve answer %q does not start with %q", text1, wantPrefix)
	}
	l.log(fmt.Sprintf("RECIPE serve RUN1 EXACT golden-prefix (%d tokens) + phrase; full answer %d tokens, first=%d e2e=%s",
		len(goldenChain), len(chain1), chain1[0], e2e1.Round(time.Millisecond)))

	// ---- determinism: second serve must reproduce the chain bit-for-bit ---
	chain2, text2, e2e2, _, err := serveOnce("RUN2")
	if err != nil {
		return err
	}
	if err := requireIntSliceEqual("recipe serve determinism chain", chain2, chain1); err != nil {
		return err
	}
	if text2 != text1 {
		return fmt.Errorf("recipe serve RUN2 answer %q != RUN1 %q", text2, text1)
	}
	l.log(fmt.Sprintf("RECIPE serve DETERMINISTIC RUN1==RUN2 chain (%d tokens) e2e1=%s e2e2=%s",
		len(chain1), e2e1.Round(time.Millisecond), e2e2.Round(time.Millisecond)))
	l.log(fmt.Sprintf("RECIPE serve ANSWER %q", text1))
	l.log("RECIPE serve LANE GREEN (served THROUGH activated recipe " + recipeID + ")")
	return nil
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}
