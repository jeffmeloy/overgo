package main

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
	"overgo/internal/patchtower"
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

// readEOSTokenIDs: scalar or list generation stop tokens.
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

// openRecipeStore: writable workflow lineage store.
func openRecipeStore(repo string) (*repodb.Store, error) {
	repository := repo
	if repository == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return nil, err
		}
		repository = roots.Store
	}
	return repodb.Open(repository)
}

// runRecipeServe: canonical case through the active recipe.
func runRecipeServe(l *ladder, repo, imagePath, question string) error {
	ctx := context.Background()

	store, err := openRecipeStore(repo)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID, program, err := resolveActiveVQA(ctx, store, l.modelDir)
	if err != nil {
		return err
	}
	recipeID := program.Definition().ID.String()
	l.log(fmt.Sprintf("RECIPE serve active vqa recipe resolved model=%s recipe=%s", modelID, recipeID))

	if _, statErr := os.Stat(imagePath); statErr != nil {
		return fmt.Errorf("serve image not found: %s", imagePath)
	}
	rawImg, err := os.ReadFile(imagePath)
	if err != nil {
		return err
	}

	// Golden chain: required prefix; serve continues to EOS.
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}
	goldenChain := dg.GeneratedTokens
	const wantPrefix = "The stovetop holds a metal pot on the left burner"
	const maxSteps = 64

	serveOnce := func(
		executeContext context.Context,
		inputImage []byte,
		inputQuestion string,
		tag string,
	) ([]int, string, time.Duration, fullResult, error) {
		pre, err := patchtower.LoadPreprocessConfig(l.modelDir)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		tok, err := hfbpe.Load(l.modelDir)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		specials, err := routedlm.LoadPromptSpecials(l.modelDir, promptRoles)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		rgb, height, width, err := patchtower.DecodeImageBytesRGB(inputImage)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		pixelValues, gridT, gridH, gridW, err := patchtower.PreprocessImage(pre, rgb, height, width)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		inputIDs, err := routedlm.RenderVisionQAPrompt(tok, specials, inputQuestion, gridH, gridW, pre.MergeSize)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
		positions := routedlm.ImageMaskPositions(inputIDs, specials)
		l.log(fmt.Sprintf("RECIPE serve %s PROCESSOR image=%s grid=[%d,%d,%d] promptLen=%d imageTokens=%d question=%q",
			tag, filepath.Base(imagePath), gridT, gridH, gridW, len(inputIDs), len(positions), inputQuestion))
		eosIDs, err := readEOSTokenIDs(l.modelDir)
		if err != nil {
			return nil, "", 0, fullResult{}, err
		}
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
		res, err := runFullPipeline(l, executeContext, worker, exe, pc, pixelValues, fullOpts{maxSteps: maxSteps, eosIDs: eosIDs})
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

	executeRecipe := func(tag string) ([]int, string, time.Duration, fullResult, error) {
		var chain []int
		var wall time.Duration
		var result fullResult
		var pipelineErr error
		answer, err := executeVQA(
			ctx, store, modelID, program, "recipe/vqa/"+strings.ToLower(tag), rawImg, question,
			func(executeContext context.Context, image []byte, question string) (string, error) {
				var text string
				chain, text, wall, result, pipelineErr = serveOnce(executeContext, image, question, tag)
				return text, pipelineErr
			},
		)
		return chain, answer, wall, result, err
	}

	chain1, text1, e2e1, _, err := executeRecipe("RUN1")
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

	// Second pass: bit-exact determinism.
	chain2, text2, e2e2, _, err := executeRecipe("RUN2")
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
