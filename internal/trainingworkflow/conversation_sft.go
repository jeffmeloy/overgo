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
	conversationEvidence    = "conversations rendered through the model's chat template, each one record, judged by next-token accuracy on held-out conversations"
)

// ConversationRenderer renders one conversation as the text the model's chat
// template makes of it, the way serving renders a prompt.
type ConversationRenderer func([]dataset.ChatMessage) (string, error)

// ConversationSFT fine-tunes a text model on a registered conversation
// dataset: one JSON file per conversation, each an OpenAI messages record.
type ConversationSFT struct {
	Dataset artifact.ID
	Render  ConversationRenderer
	// ModelDirectory is the model to train: a GGUF file or a safetensors
	// directory. OutputDirectory receives the trained model.
	ModelDirectory  string
	OutputDirectory string
	// WorkDirectory receives the rendered conversations, one text file each.
	WorkDirectory string
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
// train, never the serving ones that call this. The trainer has no loss mask,
// so it learns every token of a rendered conversation, the user's and the
// template's as well as the assistant's.
func RunConversationSFT(ctx context.Context, store *overgodb.Store, request ConversationSFT) (HeldoutResult, error) {
	if ctx == nil || store == nil || request.Render == nil || request.Dataset.Kind() != artifact.KindDataset || request.WorkDirectory == "" {
		return HeldoutResult{}, errors.New("training workflow: a conversation run needs a store, a conversation dataset, a renderer and a work directory")
	}
	texts, err := renderConversations(ctx, store, request.Dataset, request.Render)
	if err != nil {
		return HeldoutResult{}, err
	}
	registered, err := registerRendered(ctx, store, request.WorkDirectory, texts)
	if err != nil {
		return HeldoutResult{}, err
	}
	alias, err := publishConversationObjective(ctx, store, registered, len(texts))
	if err != nil {
		return HeldoutResult{}, err
	}
	return RunTextHeldout(ctx, store, TextHeldoutRequest{
		Alias: alias, ModelDirectory: request.ModelDirectory, OutputDirectory: request.OutputDirectory,
		Steps: request.Steps, MaximumSequence: conversationWindow, Host: true,
		Chat: true, Step: request.Step, Progress: request.Progress, LoRA: request.LoRA,
	})
}

// renderConversations reads each conversation of the dataset, in file order,
// and renders it. A conversation that renders the same as an earlier one is
// dropped: the inventory keeps one record for identical bytes. A dataset of
// fewer than two distinct conversations cannot hold one out to judge the
// others by.
func renderConversations(ctx context.Context, store *overgodb.Store, source artifact.ID, render ConversationRenderer) ([]string, error) {
	files, err := conversationFiles(ctx, store, source)
	if err != nil {
		return nil, err
	}
	var texts []string
	seen := map[string]bool{}
	for _, file := range files {
		text, err := renderConversation(file, render)
		if err != nil {
			return nil, err
		}
		if !seen[text] {
			seen[text] = true
			texts = append(texts, text)
		}
	}
	if len(texts) < conversationMinimum {
		return nil, fmt.Errorf("training workflow: %d distinct conversations: a run holds one out to judge the rest, so it needs at least %d", len(texts), conversationMinimum)
	}
	return texts, nil
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

// renderConversation reads one conversation file and renders it.
func renderConversation(file string, render ConversationRenderer) (string, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return "", err
	}
	var conversation dataset.Conversation
	if err := json.Unmarshal(data, &conversation); err != nil || len(conversation.Messages) == 0 {
		return "", fmt.Errorf("training workflow: %s is not a conversation: %w", filepath.Base(file), errors.Join(err, errors.New("no messages")))
	}
	text, err := render(conversation.Messages)
	if err != nil {
		return "", fmt.Errorf("training workflow: render %s: %w", filepath.Base(file), err)
	}
	if text == "" {
		return "", fmt.Errorf("training workflow: %s renders as no text", filepath.Base(file))
	}
	return text, nil
}

// PreviewConversations pages the conversations of a registered dataset as a
// run reads them: each rendered through the model's chat template and encoded
// with the tokenizer beside modelInput, every token marked trained because a
// run has no loss mask. The count is the dataset's conversation files; only
// the page is read, so the token lengths across the dataset are not listed.
func PreviewConversations(ctx context.Context, store *overgodb.Store, source artifact.ID, render ConversationRenderer, modelInput string, position, limit int) (Preview, error) {
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
		text, err := renderConversation(files[index], render)
		if err != nil {
			return Preview{}, err
		}
		ids, err := tokens.encode(text)
		if err != nil {
			return Preview{}, err
		}
		sequence := PreviewSequence{Label: "conversation", Tokens: make([]PreviewToken, len(ids))}
		for at, id := range ids {
			sequence.Tokens[at] = PreviewToken{ID: id, Piece: tokens.piece(id), Trained: true}
		}
		preview.Page = append(preview.Page, PreviewRecord{ID: filepath.Base(files[index]), Sequences: []PreviewSequence{sequence}})
	}
	return preview, nil
}

// registerRendered writes the texts as one file each, named in record order,
// under a directory the content names, and registers it as a dataset:
// rendering the same conversations the same way registers the same dataset.
func registerRendered(ctx context.Context, store *overgodb.Store, work string, texts []string) (dataset.RegisteredDataset, error) {
	digest := sha256.New()
	for _, text := range texts {
		fmt.Fprintf(digest, "%d:%s", len(text), text)
	}
	name := "sft-" + hex.EncodeToString(digest.Sum(nil))[:conversationNameDigits]
	directory := filepath.Join(work, name)
	if err := os.MkdirAll(directory, clioptions.OutputDirectoryMode); err != nil {
		return dataset.RegisteredDataset{}, err
	}
	width := len(strconv.Itoa(len(texts)))
	for index, text := range texts {
		if err := os.WriteFile(filepath.Join(directory, fmt.Sprintf("%0*d.txt", width, index+1)), []byte(text), clioptions.PrivateFileMode); err != nil {
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
	// Partitions list by name, the order a split canonicalizes them in.
	plan, err := dataset.SplitOf(source, 0, []dataset.PartitionRecords{
		{Name: conversationHeldout, Records: ordered[:held]}, {Name: conversationTrain, Records: ordered[held:]},
	})
	if err != nil {
		return dataset.SplitPlan{}, dataset.Membership{}, err
	}
	return plan, plan.Memberships[1], nil // the training partition, the second by name
}
