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
	"overgo/internal/optimizer"
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
	// Sampled, when set, receives the strips so far as JSON each time the run
	// has generated, between updates as Step does, so they can be shown while
	// the run goes; a round a cancel ends is not handed out.
	Sampled func(strips string)
	// CheckpointEvery, when positive, saves a checkpoint beside the output directory
	// (output-step-000040) after every that many updates but the last, and
	// CheckpointKeep of the newest of those the run saved are kept, so a long host
	// run survives a stop; a directory of that name an earlier run left is neither
	// kept nor deleted, and one the run would save to refuses the run.
	CheckpointEvery int
	CheckpointKeep  int
	// Resume names a checkpoint a run saved this way, to continue from: the run
	// takes the updates after the checkpoint's from its weights, optimizer state and
	// moving average, and ends where an uninterrupted run would. Its baseline, judged
	// before it trains, is still the base model's.
	Resume string
	// Proof re-verifies the recipe by the training run alone, judging and
	// publishing no verdict (HeldoutRun.Proof).
	Proof bool
	// LoRA, when set, trains a low-rank adapter over the frozen model rather
	// than the model: the run is a host run, its program's parameters are the
	// adapter's factors and the rate its recipe derives is for that many
	// parameters. The checkpoint holds the model with the adapter merged, which
	// the verdict judges, and the adapter beside it as a PEFT directory.
	LoRA *LoRA
	// Order, when not zero, seeds the order the training records are packed
	// in, in place of membership order: a run of fewer updates than the
	// stream holds windows then trains on other text for each seed, and two
	// runs that differ in nothing else are two changes of one base. The
	// held-out records and their order do not depend on it.
	Order uint64
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
	// An objective a conversation run published holds records of segments, each marked learned or
	// only read; trained as raw text, it would learn every token of the segments' JSON and be judged
	// by the same accuracy as a masked run, a verdict that looks equal to one and is not.
	if strings.HasPrefix(request.Alias, conversationAliasPrefix) && !request.Conversations {
		return HeldoutResult{}, fmt.Errorf("training workflow: %s is a conversation objective, whose records are segments of rendered conversations that only a conversation run reads; train it through one", request.Alias)
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
	if request.CheckpointEvery < 0 || request.CheckpointEvery > 0 && request.CheckpointKeep < 1 || (request.CheckpointEvery > 0 || request.Resume != "") && !request.Host {
		return HeldoutResult{}, errors.New("training workflow: a run that saves checkpoints, every positive number of updates and keeping at least one, or resumes from one trains on the host")
	}
	if request.SampleEvery < 0 || request.SampleEvery > 0 && !request.Conversations {
		return HeldoutResult{}, errors.New("training workflow: samples are generated after the prompts of a conversation run, every positive number of steps")
	}
	if request.SampleEvery > 0 && sampleRounds(request.Steps, request.SampleEvery) > sampleRoundsMost {
		return HeldoutResult{}, fmt.Errorf("training workflow: sampling every %d of %d updates generates %d times, and a run samples at most %d: sample every %d updates or more",
			request.SampleEvery, request.Steps, sampleRounds(request.Steps, request.SampleEvery), sampleRoundsMost, (request.Steps-1)/(sampleRoundsMost-1)+1)
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
	var model *densecausal.Model
	if request.LoRA != nil && request.LoRA.Quantized {
		var hold func(string) bool
		if hold, err = request.LoRA.frozenWeights(); err == nil {
			model, err = loadFrozenModel(request.ModelDirectory, hold)
		}
	} else {
		model, err = loadTrainableModel(request.ModelDirectory)
	}
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
	data, records, err := orderedTextObjectiveData(ctx, store, objective, tokenizer, sequence, request.Conversations, request.Order)
	if err != nil {
		return HeldoutResult{}, err
	}
	muonPlan, err := model.TrainingPlan()
	var adapter *loratrain.Trainer
	if request.LoRA != nil {
		// A new adapter leaves the model equal to the base until its first update; one that
		// continues a prior adapter applies it at load, so it is taken out again here and put
		// back when the run trains: the baseline below is judged on the base, never on the prior
		// adapter's model, which would read the continuation as base to trained. Its plan, over
		// the factors, is the one the program, the recipe's rate and the checkpoint name.
		if adapter, err = request.LoRA.trainer(model, request.ModelDirectory); err == nil {
			muonPlan = adapter.Plan()
			if request.LoRA.From != "" {
				adapter.RestoreBase()
			}
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
		sampling.sampled = request.Sampled
	}
	// A run that starts at its first update knows every name it will save to now, before its
	// baseline; one that resumes knows its first update only when it has read the checkpoint.
	if request.CheckpointEvery > 0 && request.Resume == "" {
		if err := checkStepNames(request.OutputDirectory, request.CheckpointEvery, 0, plan.Updates()); err != nil {
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
			// A resumed run takes the windows after the ones the checkpoint's updates took, from the
			// weights, optimizer state and average the checkpoint holds; its baseline, judged before
			// this, is still the base model's.
			if request.LoRA != nil && request.LoRA.From != "" && request.Resume == "" {
				if err := adapter.Reapply(); err != nil {
					return trainingprogram.Checkpoint{}, err
				}
			}
			var resumed resumedRun
			if request.Resume != "" {
				if resumed, err = restoreRun(request.Resume, modelID, objective, model, muonPlan, adapter != nil, plan.Updates(), config.BaseLearningRate, config.Momentum); err != nil {
					return trainingprogram.Checkpoint{}, err
				}
				if resumed.average != nil && request.Average != resumed.average.Decay() && request.Progress != nil {
					fmt.Fprintf(request.Progress, "resuming continues the checkpoint's moving average, at decay %g after %d updates, not the decay %g this run asked for\n",
						resumed.average.Decay(), resumed.average.Updates(), request.Average)
				}
				batches = batches[resumed.done:]
				if trained != nil {
					trained = trained[resumed.done:]
				}
				if resumed.adapter != nil {
					adapter = resumed.adapter
				}
			}
			first := resumed.done
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
				if err := sampling.sample(ctx, first); err != nil {
					return trainingprogram.Checkpoint{}, err
				}
			}
			// A run that saves checkpoints writes one after every CheckpointEvery updates, never
			// the last, which is the run's own checkpoint, and keeps the newest CheckpointKeep of
			// those it wrote. A fresh run checked its names before its baseline; a resumed one,
			// whose first update is known only now, checks them here.
			saves := &stepSaves{keep: request.CheckpointKeep}
			if request.CheckpointEvery > 0 && request.Resume != "" {
				if err := checkStepNames(request.OutputDirectory, request.CheckpointEvery, first, plan.Updates()); err != nil {
					return trainingprogram.Checkpoint{}, err
				}
			}
			// A run that holds some adapted projections quantized publishes the base as stored, the
			// adapter beside it, so the projections it merged in place publish as the base had them.
			published := func() *densecausal.Model {
				if adapter != nil {
					if unmerged := adapter.Unmerged(); len(unmerged) > 0 {
						return model.WithWeights(unmerged)
					}
				}
				return model
			}
			save := func(done int, state densecausal.TrainState, average *optimizer.Average) error {
				stream := trainingdata.StreamState{Identity: data.Training, Position: windowPosition(documents, sequence, done)}
				companions := []func(string) error{averageCompanion(average, muonPlan)}
				if adapter != nil {
					companions = append(companions, adapterCompanion(adapter.Adapter()))
				}
				target := stepCheckpointDirectory(request.OutputDirectory, done)
				if _, err := publishTrainedText(ctx, store, request, target, false, published(), muonPlan, recipeID, objective, stream, state, companions...); err != nil {
					return err
				}
				return saves.saved(target)
			}
			// The run reports each update, counted from the run's first, and stops between updates
			// once its context ends.
			observe := func(step int, loss float64, stepWall time.Duration) error {
				done := first + step + 1
				if request.Progress != nil {
					fmt.Fprintf(request.Progress, "step %d/%d loss %.6f wall %s backend %s\n",
						done, plan.Updates(), loss, stepWall.Round(time.Millisecond), lane)
				}
				if request.Step != nil {
					request.Step(done, plan.Updates(), loss)
				}
				if sampling != nil && done%request.SampleEvery == 0 && done < plan.Updates() {
					if err := sampling.sample(ctx, done); err != nil {
						return err
					}
				}
				if adapter != nil && request.CheckpointEvery > 0 && done%request.CheckpointEvery == 0 && done < plan.Updates() {
					state, err := adapter.OptimizerState()
					if err != nil {
						return err
					}
					if err := save(done, state, adapter.AverageState()); err != nil {
						return err
					}
				}
				return ctx.Err()
			}
			var state densecausal.TrainState
			var companions []func(stage string) error
			if adapter != nil {
				if adapter.AverageState() == nil && request.Average > 0 {
					err = adapter.Average(request.Average)
				}
				if err == nil {
					_, err = adapter.Train(batches, trained, config.BaseLearningRate, config.Momentum, observe)
				}
				if err == nil && adapter.AverageState() != nil {
					err = adapter.UseAverage()
				}
				if err == nil {
					state, err = adapter.OptimizerState()
				}
				companions = append(companions, adapterCompanion(adapter.Adapter()))
			} else {
				var resume *densecausal.TrainState
				if request.Resume != "" {
					resume = &resumed.state
				}
				_, _, state, err = runTrainingState(model, batches, densecausal.TrainOptions{
					Trained: trained, Average: request.Average, ResumeAverage: resumed.average,
					CheckpointEvery: request.CheckpointEvery, Checkpoint: save,
				}, config.BaseLearningRate, config.Momentum, request.Device, !request.Host, request.FreezeLexical, resume, observe)
			}
			if err == nil && sampling != nil {
				err = sampling.sample(ctx, plan.Updates())
			}
			if err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			stream := trainingdata.StreamState{Identity: data.Training, Position: position}
			return publishTrainedText(ctx, store, request, request.OutputDirectory, true, published(), muonPlan, recipeID, objective, stream, state, companions...)
		},
		Predict: func(example HeldoutExample) (HeldoutPrediction, error) {
			if generation {
				return generateAnswer(ctx, model, tokenizer.piece, records[example.Record].prompt)
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
	for index, document := range documents {
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
	return windows, flags, windowPosition(documents, sequence, updates), nil
}

// windowPosition is how many documents the first updates windows of sequence
// tokens reach in the stream packWindows cuts them from, a document counting once
// its first token entered a window: the stream position a checkpoint after that
// many updates records.
func windowPosition(documents [][]int, sequence, updates int) uint64 {
	starts := make([]int, len(documents))
	length := 0
	for index, document := range documents {
		starts[index] = length
		length += len(document) + 1
	}
	read := updates * sequence
	begun, _ := slices.BinarySearch(starts, read%length)
	return uint64(read/length*len(documents) + begun)
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
	return orderedTextObjectiveData(ctx, reader, objective, tokenizer, sequence, conversations, 0)
}

// orderedTextObjectiveData is textObjectiveData with the training records
// read in the order a seed draws (seededObjectiveExamples).
func orderedTextObjectiveData(ctx context.Context, reader artifact.Reader, objective trainingprogram.ObjectiveDocument, tokenizer trainableTokenizer, sequence int, conversations bool, order uint64) (ObjectiveData, map[string]textRecord, error) {
	generation := objective.Metric == trainingprogram.MetricGeneratedAnswer
	records := map[string]textRecord{}
	data, err := seededObjectiveExamples(ctx, reader, objective, order, []byte("overgo/training/text-utf8/v1"), recipecontract.ModalityText,
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
			target, err := nextTargets(tokens, trained)
			if err != nil {
				return HeldoutExample{}, err
			}
			return HeldoutExample{Input: dtype.ConvertSlice[float64](tokens), Target: dtype.ConvertSlice[float64](target), Answers: answers}, nil
		})
	return data, records, err
}

// publishTrainedText publishes the model a held-out text run trained under
// recipeID as a checkpoint at target, and records it when record is set: the
// program and run plan the recipe's objective compiles over the model's optimizer
// plan, its embedding and head frozen when the request froze them, its dataset,
// training membership and declared processor, and stream, the position the run
// reached in the training membership. A checkpoint a run saves along the way is
// published and not recorded, so the store holds no candidate model for it.
func publishTrainedText(
	ctx context.Context,
	store *overgodb.Store,
	request TextHeldoutRequest,
	target string,
	record bool,
	model *densecausal.Model,
	muonPlan hostoptimizer.Plan,
	recipeID artifact.ID,
	objective trainingprogram.ObjectiveDocument,
	stream trainingdata.StreamState,
	state densecausal.TrainState,
	companions ...func(stage string) error,
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
	checkpoint, err := publishCheckpoint(ctx, store, request.ModelDirectory, target, model, authority, state, companions...)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if record {
		_, err = RecordCheckpoint(ctx, store, target, checkpoint)
	}
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
func generateAnswer(ctx context.Context, model *densecausal.Model, piece func(int) string, prompt generationPrompt) (HeldoutPrediction, error) {
	tokens := slices.Clone(prompt.tokens)
	var text strings.Builder
	for range prompt.length {
		if err := ctx.Err(); err != nil {
			return HeldoutPrediction{}, err
		}
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

// nextTargets is what a record's first tokens are judged on predicting: every
// token after the first, or for a conversation, which marks the tokens it learns
// in trained, only those; a record that leaves none to predict is refused.
func nextTargets(tokens []int, trained []bool) ([]int, error) {
	if len(tokens) < 2 {
		return nil, fmt.Errorf("encodes to %d token(s); a next-token target needs two", len(tokens))
	}
	if trained == nil {
		return tokens[1:], nil
	}
	if target := learnedTargets(tokens, trained); len(target) > 0 {
		return target, nil
	}
	return nil, fmt.Errorf("learns no token within its first %d: its prompt fills them", len(tokens))
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
