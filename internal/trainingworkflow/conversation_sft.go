package trainingworkflow

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// The shape of a conversation fine-tuning run. A run holds out a tenth of the
// conversations, never more than scoring can afford: held-out scoring runs on
// the host, about 13 minutes a pass for 56 records of 512 tokens. Training
// windows are 512 tokens, the length of the measured fine-tuning runs.
const (
	conversationHeldoutShare   = 10
	conversationHeldoutRecords = 16
	conversationWindow         = 512
	// conversationMinimum is the fewest distinct conversations a run takes: one
	// held out to judge the rest by, and one to train on.
	conversationMinimum = 2
	// conversationNameDigits is how many hex digits of a digest name the
	// directory a run renders into and the alias its objective publishes under.
	conversationNameDigits = 12
	// conversationAverageDecay is the decay of the moving average a run keeps
	// when asked to, the one ai-toolkit's EMA recommends; the warm-up of
	// optimizer.Average keeps a short run from being mostly its start.
	conversationAverageDecay = 0.99
)

// Names of the partitions a conversation split holds, and of the identities
// and alias a conversation objective derives from.
const (
	conversationTrain   = "train"
	conversationHeldout = "heldout"
	conversationDerived = "overgo/conversation-sft/"
	// conversationAliasPrefix names the objectives conversation runs publish;
	// each takes its own alias so the shared text objective alias never moves.
	conversationAliasPrefix = "objective.sft."
	conversationEvidence    = "conversations rendered through the model's chat template and cut into the stretches a run learns, each one record, judged by next-token accuracy on the learned tokens of held-out conversations"
)

// ConversationRenderer renders one conversation as the text the model's chat
// template makes of it, the way serving renders a prompt: with the generation
// prompt that invites the next reply when generationPrompt is set, and ending
// on the last message as it was said when it is not.
type ConversationRenderer func(messages []dataset.ChatMessage, generationPrompt bool) (string, error)

// ConversationSFT fine-tunes a text model on a registered conversation
// dataset: one JSON file per conversation, each an OpenAI messages record.
type ConversationSFT struct {
	Dataset artifact.ID
	Render  ConversationRenderer
	// ModelDirectory is the model to train: a GGUF file or a safetensors
	// directory. OutputDirectory receives the trained model.
	ModelDirectory  string
	OutputDirectory string
	// WorkDirectory receives the rendered conversations, one file each.
	WorkDirectory string
	// Prompts, when set, learns every token of each conversation, the user's
	// turns and the template's markers as well as the assistant's replies. A
	// run learns the replies alone otherwise.
	Prompts bool
	// Average, when set, trains to a moving average of the weights, or of an
	// adapter's factors, at the decay conversationAverageDecay: the trained
	// model the run publishes and the verdict judges is the average.
	Average bool
	// SampleEvery, when positive, has the run generate after the prompts of two held-out
	// conversations before it trains, every that many updates and at its end, and publish the
	// generations (HeldoutResult.Samples) for the page to show in step order.
	SampleEvery int
	// CheckpointEvery, when positive, saves a checkpoint beside OutputDirectory after every that
	// many updates but the last, CheckpointKeep of the newest kept, and Resume names one to continue
	// from (TextHeldoutRequest).
	CheckpointEvery int
	CheckpointKeep  int
	Resume          string
	// Steps counts updates, each on one packed window of rendered text.
	Steps int
	// LoRA, when set, trains a low-rank adapter over the frozen model instead
	// of the model.
	LoRA *LoRA
	// Step, when set, reports each update as it finishes.
	Step     func(step, total int, loss float64)
	Progress io.Writer
}

// RunConversationSFT renders the dataset's conversations through the model's
// chat template, registers the text as a dataset, publishes a token-prediction
// objective over it under its own alias, holding a few conversations out, and
// trains the model on the rest, then judges it on the held-out ones. It trains
// on the host processor: the device trainer links only into the commands that
// train, never the serving ones that call this. A run learns the assistant's
// replies, as a model is served to give them, and only reads the user's turns
// and the template's markers; Prompts learns those too. The loss and the
// held-out accuracy count the learned tokens alone.
func RunConversationSFT(ctx context.Context, store *overgodb.Store, request ConversationSFT) (HeldoutResult, error) {
	if ctx == nil || store == nil || request.Render == nil || request.Dataset.Kind() != artifact.KindDataset || request.WorkDirectory == "" {
		return HeldoutResult{}, errors.New("training workflow: a conversation run needs a store, a conversation dataset, a renderer and a work directory")
	}
	records, err := conversationRecords(ctx, store, request.Dataset, request.Render, request.Prompts)
	if err != nil {
		return HeldoutResult{}, err
	}
	registered, err := registerRecords(ctx, store, request.WorkDirectory, records)
	if err != nil {
		return HeldoutResult{}, err
	}
	alias, err := publishConversationObjective(ctx, store, registered, len(records))
	if err != nil {
		return HeldoutResult{}, err
	}
	var decay float64
	if request.Average {
		decay = conversationAverageDecay
	}
	return RunTextHeldout(ctx, store, TextHeldoutRequest{
		Alias: alias, ModelDirectory: request.ModelDirectory, OutputDirectory: request.OutputDirectory,
		Steps: request.Steps, MaximumSequence: conversationWindow, Host: true,
		Chat: true, Conversations: true, Step: request.Step, Progress: request.Progress, LoRA: request.LoRA, Average: decay, SampleEvery: request.SampleEvery,
		CheckpointEvery: request.CheckpointEvery, CheckpointKeep: request.CheckpointKeep, Resume: request.Resume,
	})
}

// conversationRecords reads each conversation of the dataset, in file order,
// renders it and cuts it into segments, and encodes each as the record a run
// reads. A conversation that renders the same as an earlier one is dropped:
// the inventory keeps one record for identical bytes. A dataset of fewer than
// two distinct conversations cannot hold one out to judge the others by.
func conversationRecords(ctx context.Context, store *overgodb.Store, source artifact.ID, render ConversationRenderer, prompts bool) ([][]byte, error) {
	files, err := conversationFiles(ctx, store, source)
	if err != nil {
		return nil, err
	}
	var records [][]byte
	seen := map[string]bool{}
	for _, file := range files {
		segments, err := conversationSegments(file, render, prompts)
		if err != nil {
			return nil, err
		}
		record, err := json.Marshal(conversationRecord{Segments: segments})
		if err != nil {
			return nil, err
		}
		if !seen[string(record)] {
			seen[string(record)] = true
			records = append(records, record)
		}
	}
	if len(records) < conversationMinimum {
		return nil, fmt.Errorf("training workflow: %d distinct conversations: a run holds one out to judge the rest, so it needs at least %d", len(records), conversationMinimum)
	}
	return records, nil
}

// conversationFiles lists the conversation files of a registered dataset, in
// file order.
func conversationFiles(ctx context.Context, store *overgodb.Store, source artifact.ID) ([]string, error) {
	directory, err := artifact.AvailablePath(ctx, store, source, artifact.LocationDirectory)
	if err != nil {
		return nil, fmt.Errorf("training workflow: conversation dataset: %w", err)
	}
	files, err := filepath.Glob(filepath.Join(directory, "*.json"))
	if err != nil {
		return nil, err
	}
	slices.Sort(files)
	return files, nil
}

// conversationSegments reads one conversation file, renders it and cuts it
// into the stretches a run learns and the ones it reads.
func conversationSegments(file string, render ConversationRenderer, prompts bool) ([]conversationSegment, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var conversation dataset.Conversation
	if err := json.Unmarshal(data, &conversation); err != nil || len(conversation.Messages) == 0 {
		return nil, fmt.Errorf("training workflow: %s is not a conversation: %w", filepath.Base(file), errors.Join(err, errors.New("no messages")))
	}
	segments, err := segmentConversation(conversation.Messages, render, prompts)
	if err != nil {
		return nil, fmt.Errorf("training workflow: render %s: %w", filepath.Base(file), err)
	}
	return segments, nil
}

// PreviewConversations pages the conversations of a registered dataset as a
// run reads them: each rendered through the model's chat template and encoded
// with the tokenizer beside modelInput, each token marked trained when a run
// learns it, which is every token with prompts set and the assistant's replies
// otherwise. The count is the dataset's conversation files; only the page is
// read, so the token lengths across the dataset are not listed.
func PreviewConversations(ctx context.Context, store *overgodb.Store, source artifact.ID, render ConversationRenderer, modelInput string, prompts bool, position, limit int) (Preview, error) {
	if ctx == nil || store == nil || render == nil || source.Kind() != artifact.KindDataset || position < 0 || limit <= 0 {
		return Preview{}, errors.New("training workflow: a conversation preview needs a store, a conversation dataset, a renderer and a valid page")
	}
	files, err := conversationFiles(ctx, store, source)
	if err != nil {
		return Preview{}, err
	}
	tokens, err := loadTrainableTokenizerFor(modelInput, true)
	if err != nil {
		return Preview{}, err
	}
	preview := Preview{Objective: trainingprogram.ObjectiveTokenPrediction, Records: len(files), Position: position}
	for index := position; index < min(len(files), position+limit); index++ {
		segments, err := conversationSegments(files[index], render, prompts)
		if err != nil {
			return Preview{}, err
		}
		ids, trained, err := encodeConversation(tokens, segments)
		if err != nil {
			return Preview{}, fmt.Errorf("training workflow: %s: %w", filepath.Base(files[index]), err)
		}
		sequence := PreviewSequence{Label: "conversation", Tokens: make([]PreviewToken, len(ids))}
		for at, id := range ids {
			sequence.Tokens[at] = PreviewToken{ID: id, Piece: tokens.piece(id), Trained: trained[at]}
		}
		preview.Page = append(preview.Page, PreviewRecord{ID: filepath.Base(files[index]), Sequences: []PreviewSequence{sequence}})
	}
	return preview, nil
}

// registerRecords writes the records as one file each, named in record
// order, under a directory the content names, and registers it as a dataset:
// rendering the same conversations the same way, learning the same stretches
// of them, registers the same dataset. A record is JSON, kept in a text file
// as a generation objective's records are.
func registerRecords(ctx context.Context, store *overgodb.Store, work string, records [][]byte) (dataset.RegisteredDataset, error) {
	digest := sha256.New()
	for _, record := range records {
		fmt.Fprintf(digest, "%d:%s", len(record), record)
	}
	name := "sft-" + hex.EncodeToString(digest.Sum(nil))[:conversationNameDigits]
	directory := filepath.Join(work, name)
	if err := os.MkdirAll(directory, clioptions.OutputDirectoryMode); err != nil {
		return dataset.RegisteredDataset{}, err
	}
	width := len(strconv.Itoa(len(records)))
	for index, record := range records {
		if err := os.WriteFile(filepath.Join(directory, fmt.Sprintf("%0*d.txt", width, index+1)), record, clioptions.PrivateFileMode); err != nil {
			return dataset.RegisteredDataset{}, err
		}
	}
	return dataset.RegisterDirectoryDataset(ctx, store, name, directory)
}

// publishConversationObjective publishes the token-prediction objective over
// the registered text, its held-out conversations chosen by the order of
// their record identities' digests, and binds it to an alias of its own.
func publishConversationObjective(ctx context.Context, store *overgodb.Store, registered dataset.RegisteredDataset, count int) (string, error) {
	plan, training, err := conversationSplit(registered.Dataset, count)
	if err != nil {
		return "", err
	}
	derived := func(kind artifact.Kind, role string) (artifact.ID, error) {
		return artifact.IdentifyBytes(kind, []byte(conversationDerived+role))
	}
	var processor, loss, evaluation, evidence artifact.ID
	for target, build := range map[*artifact.ID]func() (artifact.ID, error){
		&processor:  func() (artifact.ID, error) { return derived(artifact.KindProfile, "processor") },
		&loss:       func() (artifact.ID, error) { return derived(artifact.KindProfile, "loss") },
		&evaluation: func() (artifact.ID, error) { return derived(artifact.KindProfile, "evaluation") },
		&evidence: func() (artifact.ID, error) {
			return artifact.IdentifyBytes(artifact.KindEvidence, []byte(conversationEvidence))
		},
	} {
		if *target, err = build(); err != nil {
			return "", err
		}
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "conversation-sft", Kind: trainingprogram.ObjectiveTokenPrediction, Signature: textSignature,
		Dataset: registered.Dataset, Split: training.ID, Processors: []artifact.ID{processor},
		Loss: loss, Evaluation: evaluation, Metric: trainingprogram.MetricTokenAccuracy,
		HeldoutRecords: conversationHeldoutRecords, Evidence: []artifact.ID{evidence},
		// Declared: the run measures the objective; it is not asserted here.
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		return "", err
	}
	alias := conversationAliasPrefix + registered.Dataset.DigestHex()[:conversationNameDigits]
	// The same conversations rendered the same way make the same objective, which the
	// first run published with its alias.
	if _, held, err := store.Artifact(ctx, objective.ID); err != nil {
		return "", err
	} else if held {
		return alias, nil
	}
	content, err := objective.Content()
	if err != nil {
		return "", err
	}
	batch, err := plan.PublicationBatch("conversation-sft/"+objective.ID.String(), nil)
	if err != nil {
		return "", err
	}
	for _, id := range []artifact.ID{processor, loss, evaluation, evidence} {
		if _, held, err := store.Artifact(ctx, id); err != nil {
			return "", err
		} else if !held {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
		}
	}
	batch.Contents = append(batch.Contents, content)
	if move, moved, err := artifact.MoveAlias(ctx, store, alias, objective.ID); err != nil {
		return "", err
	} else if moved {
		batch.Aliases = []artifact.AliasBinding{move}
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return "", err
	}
	return alias, nil
}

// conversationSplit divides the dataset's records into a training and a
// held-out membership: the held-out ones are the first by the digest of the
// record identity, so a dataset always splits the same way and neither side
// is ever empty.
func conversationSplit(source artifact.ID, count int) (dataset.SplitPlan, dataset.Membership, error) {
	records := make([]dataset.Record, count)
	for index := range records {
		id := dataset.RecordID(source, dataset.InventoryAssetName, uint64(index))
		records[index] = dataset.Record{ID: id, Group: id}
	}
	ordered := slices.Clone(records)
	slices.SortFunc(ordered, func(left, right dataset.Record) int {
		leftDigest, rightDigest := sha256.Sum256([]byte(left.ID)), sha256.Sum256([]byte(right.ID))
		return bytes.Compare(leftDigest[:], rightDigest[:])
	})
	held := min(max(count/conversationHeldoutShare, 1), conversationHeldoutRecords)
	var plan dataset.SplitPlan
	var training dataset.Membership
	var partitions []dataset.Partition
	// Partitions list by name, the order a split canonicalizes them in.
	for _, part := range []struct {
		name    string
		records []dataset.Record
	}{{conversationHeldout, ordered[:held]}, {conversationTrain, ordered[held:]}} {
		membership, err := dataset.NewMembership(source, 0, part.name, part.records)
		if err != nil {
			return dataset.SplitPlan{}, dataset.Membership{}, err
		}
		view, err := dataset.NewView(source, &membership.ID, nil)
		if err != nil {
			return dataset.SplitPlan{}, dataset.Membership{}, err
		}
		plan.Memberships, plan.Views = append(plan.Memberships, membership), append(plan.Views, view)
		partitions = append(partitions, dataset.Partition{Name: part.name, View: view.ID})
		if part.name == conversationTrain {
			training = membership
		}
	}
	var err error
	plan.Split, err = dataset.NewSplit(source, partitions)
	return plan, training, err
}
