package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/hostoptimizer"
	"overgo/internal/loratrain"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TextHeldoutRequest trains a text model under a registered token-prediction
// objective and judges it on the objective's held-out split.
type TextHeldoutRequest struct {
	Alias string
	// Recipe selects an exact published training recipe; zero bootstraps the objective's default.
	Recipe         artifact.ID
	ModelDirectory string
	// OutputDirectory receives the trained model as a checkpoint, the
	// trained result the verdict binds: a GGUF base's trained GGUF, else
	// its trained tensors.
	OutputDirectory string
	// Steps counts updates, each on one packed training window.
	Steps int
	// MaximumSequence is each training window's length and bounds each
	// held-out record's tokens; zero uses the model's context length.
	MaximumSequence int
	Host            bool
	// Device trains the model when Host is false.
	Device DeviceTrainer
	// FreezeLexical leaves the token embedding and output head untrained,
	// as only the device trainer does, and the run records the plan with
	// them frozen.
	FreezeLexical    bool
	MaxProjectedWall time.Duration
	// Progress, when set, receives the lane the run trains on and a line
	// per update, as the dense workflow reports them.
	Progress io.Writer
	// Step, when set, reports each update as it finishes: its number, the
	// run's total and its loss.
	Step func(step, total int, loss float64)
	// Chat reads each record as rendered chat text: the model's control
	// tokens in it parse as tokens and its start token is added, as serving
	// encodes a chat prompt.
	Chat bool
	// Conversations reads each record as a conversation cut into segments, each
	// marked learned or only read (RunConversationSFT writes them). The loss
	// counts the learned tokens alone and the held-out accuracy judges the
	// model on predicting them, a run on the host processor as only it masks.
	Conversations bool
	// Average, when positive, is the decay of a moving average of the trained
	// weights, or of an adapter's factors, taken after each update (see
	// optimizer.Average): the trained model the run publishes and the verdict
	// judges is the average, as ai-toolkit saves its EMA. Only the host trainer
	// takes it, and a run's weights alone are averaged, not its optimizer state.
	Average float64
	// SampleEvery, when positive, has a conversation run generate after the
	// first prompts of its held-out conversations before it trains, after every
	// that many updates and at its end, and publish the generations as a
	// document the result names (HeldoutResult.Samples).
	SampleEvery int
	// Proof re-verifies the recipe by the training run alone, judging and
	// publishing no verdict (HeldoutRun.Proof).
	Proof bool
	// LoRA, when set, trains a low-rank adapter over the frozen model rather
	// than the model: the run is a host run, its program's parameters are the
	// adapter's factors and the rate its recipe derives is for that many
	// parameters. The checkpoint holds the model with the adapter merged, which
	// the verdict judges, and the adapter beside it as a PEFT directory.
	LoRA *LoRA
}

// promptAnswers is one record of a generation objective: a prompt and the
// answers a generated continuation is accepted as. Training reads the
// prompt followed by the first answer.
type promptAnswers struct {
	Prompt  string   `json:"prompt"`
	Answers []string `json:"answers"`
}

// generationPrompt is what a held-out generation answers from: the
// prompt's tokens, and how many tokens to generate, as many as the longest
// accepted answer takes after the prompt, so length is not what is judged.
type generationPrompt struct {
	tokens []int
	length int
}

// textRecord is what a text run reads of one record beyond its example:
// every token of the record's training sequence, which training packs into
// windows, and, for a generation record, what a held-out generation
// answers it from.
type textRecord struct {
	tokens []int
	// trained marks the tokens a conversation record learns, one flag for each
	// of tokens; it is nil for a record that learns them all.
	trained []bool
	prompt  generationPrompt
}

// RunTextHeldout trains the model on fixed windows packed from the training
// records and judges it on the held-out records by the objective's metric.
// Training joins the training records' token sequences, in membership
// order, each followed by the model's end-of-sequence token, and cuts the
// stream into windows of the run's sequence length, one window per update,
// as language-model training reads text; the held-out records are never
// packed. Token accuracy reads each record as one UTF-8 document and asks,
// at each position of a held-out record, whether the model's most likely
// next token is the one that follows. Generated answer reads each record as
// a JSON prompt and its accepted answers, trains on the prompt and its
// first answer, and asks whether the text the model generates greedily
// after a held-out prompt is an accepted answer. The model's weights update
// in place, so the base and trained models answer in one process; the
// trained model publishes at the request's output directory before it
// answers.
func RunTextHeldout(ctx context.Context, store *overgodb.Store, request TextHeldoutRequest) (HeldoutResult, error) {
	if request.OutputDirectory == "" {
		return HeldoutResult{}, errors.New("training workflow: a held-out text run publishes its trained model, so it needs an output directory")
	}
	if request.Host && request.FreezeLexical {
		return HeldoutResult{}, errors.New("training workflow: host and frozen lexical execution are incompatible")
	}
	if request.LoRA != nil && (!request.Host || request.FreezeLexical || request.Device != nil) {
		return HeldoutResult{}, errors.New("training workflow: a LoRA run trains on the host and leaves the lexical table to the frozen base")
	}
	if request.Average != 0 && (!request.Host || !(request.Average > 0 && request.Average < 1)) {
		return HeldoutResult{}, errors.New("training workflow: a moving average of the weights is taken by the host trainer, at a decay in (0, 1)")
	}
	if request.Conversations && !request.Host {
		return HeldoutResult{}, errors.New("training workflow: a conversation run learns only some tokens, which only the host trainer does")
	}
	if request.SampleEvery < 0 || request.SampleEvery > 0 && !request.Conversations {
		return HeldoutResult{}, errors.New("training workflow: samples are generated after the prompts of a conversation run, every positive number of steps")
	}
	weights, err := IdentifyModelWeights(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	modelID, objective, session, err := resolveHeldoutAuthority(ctx, store, weights, request.Alias, request.Recipe)
	if err != nil {
		return HeldoutResult{}, err
	}
	generation := objective.Metric == trainingprogram.MetricGeneratedAnswer
	if objective.Kind != trainingprogram.ObjectiveTokenPrediction || objective.Metric != trainingprogram.MetricTokenAccuracy && !generation {
		return HeldoutResult{}, fmt.Errorf("training workflow: objective %q is %q judged by %q, not token prediction by token accuracy or generated answer", request.Alias, objective.Kind, objective.Metric)
	}
	if request.Conversations && generation {
		return HeldoutResult{}, errors.New("training workflow: conversation records are judged by token accuracy, not generated answers")
	}
	// The load is the model, its tokenizer, the objective's records and the
	// plan they train under, as the dense workflow's is.
	loadStarted := time.Now()
	model, err := loadTrainableModel(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	tokenizer, err := loadTrainableTokenizerFor(request.ModelDirectory, request.Chat)
	if err != nil {
		return HeldoutResult{}, err
	}
	if !tokenizer.ended {
		return HeldoutResult{}, errors.New("training workflow: the model declares no end-of-sequence token to separate the training records packed into one window")
	}
	sequence := model.Dims.ContextLength
	if request.MaximumSequence > 0 {
		sequence = min(sequence, request.MaximumSequence)
	}
	data, records, err := textObjectiveData(ctx, store, objective, tokenizer, sequence, request.Conversations)
	if err != nil {
		return HeldoutResult{}, err
	}
	muonPlan, err := model.TrainingPlan()
	var adapter *loratrain.Trainer
	if request.LoRA != nil {
		// The trainer leaves the model equal to the base until its first update, so the
		// base's held-out answers below are the base's. Its plan, over the factors, is
		// the one the program, the recipe's rate and the checkpoint name.
		if adapter, err = request.LoRA.trainer(model, request.ModelDirectory); err == nil {
			muonPlan = adapter.Plan()
		}
	}
	if err != nil {
		return HeldoutResult{}, err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction, Updates: request.Steps, MaximumSequence: sequence,
		MaxProjectedWall: request.MaxProjectedWall,
		Parameters:       muonPlan.ParameterCount(), Optimizer: session.Optimizer,
	})
	if err != nil {
		return HeldoutResult{}, err
	}
	config := plan.Optimizer()
	var sampling *sampler
	if request.SampleEvery > 0 {
		_, heldout, err := objectiveMemberships(ctx, store, objective)
		if err != nil {
			return HeldoutResult{}, err
		}
		ids := make([]string, len(heldout.Records))
		for index, record := range heldout.Records {
			ids[index] = record.ID
		}
		if sampling, err = newSampler(model, tokenizer.piece, ids, records, sequence); err != nil {
			return HeldoutResult{}, err
		}
	}
	result, err := runHeldout(ctx, store, HeldoutRun{
		Data: data, Load: time.Since(loadStarted), Host: request.Host, Proof: request.Proof,
		Train: func(ctx context.Context, recipeID artifact.ID, shown []HeldoutExample) (trainingprogram.Checkpoint, error) {
			documents := make([][]int, len(shown))
			var flags [][]bool
			if request.Conversations {
				flags = make([][]bool, len(shown))
			}
			for index, example := range shown {
				documents[index] = records[example.Record].tokens
				if flags != nil {
					flags[index] = records[example.Record].trained
				}
			}
			batches, trained, position, err := packWindows(documents, flags, tokenizer.end, sequence, plan.Updates())
			if err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			lane := "cuda-resident"
			if request.Host {
				lane = "host-reference"
			}
			if request.Progress != nil {
				fmt.Fprintf(request.Progress, "lane %s: %d steps, %d batches, frozen_lexical=%t\n",
					lane, plan.Updates(), len(batches), request.FreezeLexical)
			}
			// A sampled run generates once before it trains, every SampleEvery updates
			// and once more at its end, which is the average when it keeps one.
			if sampling != nil {
				if err := sampling.sample(0); err != nil {
					return trainingprogram.Checkpoint{}, err
				}
			}
			// The run reports each update and stops between updates once
			// its context ends.
			observe := func(step int, loss float64, stepWall time.Duration) error {
				if request.Progress != nil {
					fmt.Fprintf(request.Progress, "step %d/%d loss %.6f wall %s backend %s\n",
						step+1, plan.Updates(), loss, stepWall.Round(time.Millisecond), lane)
				}
				if request.Step != nil {
					request.Step(step+1, plan.Updates(), loss)
				}
				if sampling != nil && (step+1)%request.SampleEvery == 0 && step+1 < plan.Updates() {
					if err := sampling.sample(step + 1); err != nil {
						return err
					}
				}
				return ctx.Err()
			}
			var state densecausal.TrainState
			var companion func(stage string) error
			if adapter != nil {
				if request.Average > 0 {
					err = adapter.Average(request.Average)
				}
				if err == nil {
					_, err = adapter.Train(batches, trained, config.BaseLearningRate, config.Momentum, observe)
				}
				if err == nil && request.Average > 0 {
					err = adapter.UseAverage()
				}
				if err == nil {
					state, err = adapter.OptimizerState()
				}
				companion = adapterCompanion(adapter.Adapter())
			} else {
				_, _, state, err = runTrainingState(model, batches, densecausal.TrainOptions{Trained: trained, Average: request.Average}, config.BaseLearningRate, config.Momentum, request.Device, !request.Host, request.FreezeLexical, nil, observe)
			}
			if err == nil && sampling != nil {
				err = sampling.sample(plan.Updates())
			}
			if err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			stream := trainingdata.StreamState{Identity: data.Training, Position: position}
			return publishTrainedText(ctx, store, request, model, muonPlan, recipeID, objective, stream, state, companion)
		},
		Predict: func(example HeldoutExample) (HeldoutPrediction, error) {
			if generation {
				return generateAnswer(model, tokenizer.piece, records[example.Record].prompt)
			}
			return nextTokens(model, example, records[example.Record].trained)
		},
	}, modelID, objective, session.Program.Definition().ID)
	if err == nil && sampling != nil {
		result.Samples, err = sampling.publish(ctx, store, result.Trained)
	}
	return result, err
}

// packWindows joins the documents, in order, each followed by the token
// end, into one stream and cuts updates windows of sequence tokens from it.
// The stream cycles as a training stream cycles its records: a window that
// reaches the stream's end continues from its start, past the last
// document's end token, so no token is dropped, nothing is padded (the
// device trainer has no loss mask to exclude padding with) and every window is
// exactly sequence tokens, even when the whole stream is shorter than one.
// trained, when it is not nil, holds each document's flags for the tokens a
// run learns, and each window comes back with its flags: the end token that
// follows a document is learned only when every token of the document is, as
// it is a joint of the packing and not a word of any conversation, and a
// window that learns no token is refused, as nothing in it could be judged.
// It also returns how many documents the windows reached, a document
// counting once its first token entered a window, the stream position a
// checkpoint records.
func packWindows(documents [][]int, trained [][]bool, end, sequence, updates int) ([][]int, [][]bool, uint64, error) {
	if len(documents) == 0 {
		return nil, nil, 0, errors.New("training workflow: the training membership holds no text")
	}
	var stream []int
	var learned []bool
	starts := make([]int, len(documents))
	for index, document := range documents {
		starts[index] = len(stream)
		stream = append(append(stream, document...), end)
		if trained != nil {
			learned = append(append(learned, trained[index]...), !slices.Contains(trained[index], false))
		}
	}
	windows := make([][]int, updates)
	var flags [][]bool
	if trained != nil {
		flags = make([][]bool, updates)
	}
	for step := range windows {
		windows[step] = make([]int, sequence)
		if flags != nil {
			flags[step] = make([]bool, sequence)
		}
		for index := range windows[step] {
			at := (step*sequence + index) % len(stream)
			windows[step][index] = stream[at]
			if flags != nil {
				flags[step][index] = learned[at]
			}
		}
		// The first token of a window is predicted by nothing: its flag is not read.
		if flags != nil && !slices.Contains(flags[step][1:], true) {
			return nil, nil, 0, fmt.Errorf("training workflow: window %d of %d tokens holds no token to learn: prompts fill it, so the run's window is shorter than the prompt that precedes a reply", step+1, sequence)
		}
	}
	read := updates * sequence
	begun, _ := slices.BinarySearch(starts, read%len(stream))
	return windows, flags, uint64(read/len(stream)*len(documents) + begun), nil
}

// textObjectiveData reads both of a text objective's memberships as the
// held-out text run trains and judges them. Each record is one UTF-8
// document; its example is its first sequence tokens: the input is the
// sequence, the target every token after the first. A generation record's
// document is its prompt and answers, its sequence the prompt followed by
// the first answer. A conversation record is instead the segments of a
// rendered conversation, and its example's target is the tokens it learns
// alone, the ones a held-out run is judged on predicting. Every token of each
// record's sequence, which training packs, and what a held-out generation
// answers it from return keyed by record.
func textObjectiveData(ctx context.Context, reader artifact.Reader, objective trainingprogram.ObjectiveDocument, tokenizer trainableTokenizer, sequence int, conversations bool) (ObjectiveData, map[string]textRecord, error) {
	generation := objective.Metric == trainingprogram.MetricGeneratedAnswer
	records := map[string]textRecord{}
	data, err := ObjectiveExamples(ctx, reader, objective, []byte("overgo/training/text-utf8/v1"), recipecontract.ModalityText,
		trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
		func(example trainingdata.Example) (HeldoutExample, error) {
			if len(example.Values) != 1 {
				return HeldoutExample{}, errors.New("not one text document")
			}
			text, answers, prompt := string(example.Values[0].Data), []string(nil), generationPrompt{}
			if generation {
				var record promptAnswers
				var err error
				if record, prompt, err = readPromptAnswers(tokenizer.encode, example.Values[0].Data); err != nil {
					return HeldoutExample{}, err
				}
				text, answers = record.Prompt+record.Answers[0], record.Answers
			}
			var tokens []int
			var trained []bool
			var err error
			if conversations {
				var record conversationRecord
				if err = strictjson.DecodeBytes(example.Values[0].Data, &record); err == nil {
					tokens, trained, err = encodeConversation(tokenizer, record.Segments)
				}
			} else {
				tokens, err = tokenizer.encode(text)
			}
			if err != nil {
				return HeldoutExample{}, err
			}
			records[example.ID] = textRecord{tokens: tokens, trained: trained, prompt: prompt}
			tokens = tokens[:min(len(tokens), sequence)]
			if len(tokens) < 2 {
				return HeldoutExample{}, fmt.Errorf("encodes to %d token(s); a next-token target needs two", len(tokens))
			}
			target := tokens[1:]
			if trained != nil {
				if target = learnedTargets(tokens, trained); len(target) == 0 {
					return HeldoutExample{}, fmt.Errorf("learns no token within its first %d: its prompt fills them", len(tokens))
				}
			}
			return HeldoutExample{Input: dtype.ConvertSlice[float64](tokens), Target: dtype.ConvertSlice[float64](target), Answers: answers}, nil
		})
	return data, records, err
}

// publishTrainedText publishes the model a held-out text run trained under
// recipeID as a checkpoint at the request's output directory and records
// it: the program and run plan the recipe's objective compiles over the
// model's optimizer plan, its embedding and head frozen when the request
// froze them, its dataset, training membership and declared processor, and
// stream, the position the run reached in the training membership.
func publishTrainedText(
	ctx context.Context,
	store *overgodb.Store,
	request TextHeldoutRequest,
	model *densecausal.Model,
	muonPlan hostoptimizer.Plan,
	recipeID artifact.ID,
	objective trainingprogram.ObjectiveDocument,
	stream trainingdata.StreamState,
	state densecausal.TrainState,
	companion func(stage string) error,
) (trainingprogram.Checkpoint, error) {
	if len(objective.Processors) != 1 {
		return trainingprogram.Checkpoint{}, fmt.Errorf("training workflow: objective %q declares %d processors; a text checkpoint records one", objective.Name, len(objective.Processors))
	}
	if request.FreezeLexical {
		var err error
		if muonPlan, state, err = freezeLexical(model, muonPlan, state); err != nil {
			return trainingprogram.Checkpoint{}, err
		}
	}
	modelID, err := identifyModel(request.ModelDirectory)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	session, err := ResolveSessionAuthority(ctx, store, modelID, recipeID)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	authority, err := compileAuthority(ctx, store, session.Program, muonPlan, request.ModelDirectory,
		batchAuthority{Dataset: objective.Dataset, Split: objective.Split, Processor: objective.Processors[0]}, stream,
		trainingprogram.Checkpoint{}, trainingprogram.ObjectiveTokenPrediction, nil, nil)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	checkpoint, err := publishCheckpoint(ctx, store, request.ModelDirectory, request.OutputDirectory, model, authority, state, companion)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	_, err = RecordCheckpoint(ctx, store, request.OutputDirectory, checkpoint)
	return checkpoint, err
}

// freezeLexical restates a frozen-lexical run's trained state under the
// plan the run stepped: the model's Muon plan with the token embedding and
// output head frozen, as the device trainer leaves them, so the program,
// run plan and checkpoint the run records name them untrainable and differ
// from an unfrozen run's.
func freezeLexical(model *densecausal.Model, plan hostoptimizer.Plan, state densecausal.TrainState) (hostoptimizer.Plan, densecausal.TrainState, error) {
	if err := hostoptimizer.ValidateState(state, plan.Identity(), plan.ParameterCount()); err != nil {
		return hostoptimizer.Plan{}, densecausal.TrainState{}, err
	}
	embedding, _, head := model.LexicalTensorNames()
	specs := make([]hostoptimizer.GroupSpec, plan.GroupCount())
	for index, group := range plan.Groups() {
		specs[index] = group.GroupSpec
		specs[index].Frozen = group.Name == embedding || group.Name == head
	}
	frozen, err := hostoptimizer.CompilePlan(plan.ParameterCount(), specs)
	if err != nil {
		return hostoptimizer.Plan{}, densecausal.TrainState{}, err
	}
	state.PlanIdentity = frozen.Identity()
	return frozen, state, nil
}

// readPromptAnswers reads one generation record and what a held-out
// generation answers it from.
func readPromptAnswers(encode func(string) ([]int, error), data []byte) (promptAnswers, generationPrompt, error) {
	var record promptAnswers
	if err := strictjson.DecodeBytes(data, &record); err != nil {
		return promptAnswers{}, generationPrompt{}, err
	}
	if record.Prompt == "" || len(record.Answers) == 0 || slices.Contains(record.Answers, "") {
		return promptAnswers{}, generationPrompt{}, errors.New("a generation record needs a prompt and nonempty answers")
	}
	tokens, err := encode(record.Prompt)
	if err != nil {
		return promptAnswers{}, generationPrompt{}, err
	}
	generation := generationPrompt{tokens: tokens}
	for _, answer := range record.Answers {
		answered, err := encode(record.Prompt + answer)
		if err != nil {
			return promptAnswers{}, generationPrompt{}, err
		}
		generation.length = max(generation.length, len(answered)-len(tokens))
	}
	return record, generation, nil
}

// generateAnswer continues a prompt greedily, the model's most likely token
// at a time, and answers with the continuation's text.
func generateAnswer(model *densecausal.Model, piece func(int) string, prompt generationPrompt) (HeldoutPrediction, error) {
	tokens := slices.Clone(prompt.tokens)
	var text strings.Builder
	for range prompt.length {
		logits, err := model.Logits(tokens)
		if err != nil {
			return HeldoutPrediction{}, err
		}
		last := logits[(len(tokens)-1)*model.Dims.Vocab:]
		next := slices.Index(last, slices.Max(last))
		tokens = append(tokens, next)
		text.WriteString(piece(next))
	}
	return HeldoutPrediction{Text: text.String()}, nil
}

// learnedTargets are the tokens after the first that trained marks, in order:
// what a conversation record's model is judged on predicting.
func learnedTargets(tokens []int, trained []bool) []int {
	var targets []int
	for position := 1; position < len(tokens); position++ {
		if trained[position] {
			targets = append(targets, tokens[position])
		}
	}
	return targets
}

// nextTokens answers a record with the model's most likely next token at
// each position but the last, teacher-forced on the record's own tokens; a
// record with trained flags is answered at the positions whose next token
// it learns alone, which are the targets learnedTargets lists.
func nextTokens(model *densecausal.Model, example HeldoutExample, trained []bool) (HeldoutPrediction, error) {
	tokens := dtype.ConvertSlice[int](example.Input)
	_, logits, err := model.Loss(tokens)
	if err != nil {
		return HeldoutPrediction{}, err
	}
	vocabulary := model.Dims.Vocab
	predicted := make([]float64, 0, len(tokens)-1)
	for position := range len(tokens) - 1 {
		if trained != nil && !trained[position+1] {
			continue
		}
		row := logits[position*vocabulary : (position+1)*vocabulary]
		predicted = append(predicted, float64(slices.Index(row, slices.Max(row))))
	}
	return HeldoutPrediction{Values: predicted}, nil
}
