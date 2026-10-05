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
	"maps"
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
	if request.Prompts && request.SampleEvery > 0 {
		return HeldoutResult{}, errors.New("training workflow: samples are generated after a conversation's prompt, and a run that learns the prompts too has no stretch to generate after")
	}
	tokens, err := loadTrainableTokenizerFor(request.ModelDirectory, true)
	if err != nil {
		return HeldoutResult{}, err
	}
	records, skipped, err := conversationRecords(ctx, store, request.Dataset, request.Render, request.Prompts, tokens)
	if err != nil {
		return HeldoutResult{}, err
	}
	if len(skipped) > 0 && request.Progress != nil {
		fmt.Fprintf(request.Progress, "skipped %d of %d conversations the run cannot learn from; the first, %s: %s\n",
			len(skipped), len(skipped)+len(records), skipped[0].File, skipped[0].Reason)
	}
	registered, err := registerRecords(ctx, store, request.WorkDirectory, records)
	if err != nil {
		return HeldoutResult{}, err
	}
	groups, err := conversationGroups(records)
	if err != nil {
		return HeldoutResult{}, err
	}
	alias, err := publishConversationObjective(ctx, store, registered, groups)
	if err != nil {
		return HeldoutResult{}, err
	}
	var decay float64
	if request.Average {
		decay = conversationAverageDecay
	}
	result, err := RunTextHeldout(ctx, store, TextHeldoutRequest{
		Alias: alias, ModelDirectory: request.ModelDirectory, OutputDirectory: request.OutputDirectory,
		Steps: request.Steps, MaximumSequence: conversationWindow, Host: true,
		Chat: true, Conversations: true, Step: request.Step, Progress: request.Progress, LoRA: request.LoRA, Average: decay, SampleEvery: request.SampleEvery,
		CheckpointEvery: request.CheckpointEvery, CheckpointKeep: request.CheckpointKeep, Resume: request.Resume,
	})
	result.Skipped = skipped
	return result, err
}

// conversationRecords reads each conversation of the dataset, in file order,
// renders it and cuts it into segments, and encodes each as the record a run
// reads. A conversation that renders the same as an earlier one is dropped:
// the inventory keeps one record for identical bytes. A conversation the model's
// template or tokenizer cannot cut into a stretch to learn and the stretches
// before it, or whose reply the window does not reach, is skipped and returned
// with its reason, so one such record does not end the run; the records are the
// ones that remain, and a dataset of fewer than two distinct ones cannot hold
// one out to judge the others by.
func conversationRecords(ctx context.Context, store *overgodb.Store, source artifact.ID, render ConversationRenderer, prompts bool, tokens trainableTokenizer) ([][]byte, []SkippedConversation, error) {
	files, err := conversationFiles(ctx, store, source)
	if err != nil {
		return nil, nil, err
	}
	var records [][]byte
	var skipped []SkippedConversation
	seen := map[string]bool{}
	for _, file := range files {
		segments, err := conversationSegments(file, render, prompts)
		if err == nil {
			err = learnable(tokens, segments)
		}
		var unusable unusableConversation
		if errors.As(err, &unusable) {
			skipped = append(skipped, SkippedConversation{File: filepath.Base(file), Reason: err.Error()})
			continue
		}
		if err != nil {
			return nil, nil, err
		}
		record, err := json.Marshal(conversationRecord{Segments: segments})
		if err != nil {
			return nil, nil, err
		}
		if !seen[string(record)] {
			seen[string(record)] = true
			records = append(records, record)
		}
	}
	if len(records) < conversationMinimum {
		reason := ""
		if len(skipped) > 0 {
			reason = fmt.Sprintf("; the %d skipped as unusable, the first %s: %s", len(skipped), skipped[0].File, skipped[0].Reason)
		}
		return nil, nil, fmt.Errorf("training workflow: %d distinct conversations: a run holds one out to judge the rest, so it needs at least %d%s", len(records), conversationMinimum, reason)
	}
	return records, skipped, nil
}

// learnable holds a rendered conversation to what a run needs of its record under
// the model's tokenizer: tokens that tell a reply from its prompt, and a token to
// learn within the window.
func learnable(tokens trainableTokenizer, segments []conversationSegment) error {
	ids, trained, err := encodeConversation(tokens, segments)
	if err == nil {
		_, err = nextTargets(ids[:min(len(ids), conversationWindow)], trained)
	}
	if err != nil {
		return unusableConversation{err}
	}
	return nil
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
		return nil, unusableConversation{fmt.Errorf("training workflow: render %s: %w", filepath.Base(file), err)}
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
func publishConversationObjective(ctx context.Context, store *overgodb.Store, registered dataset.RegisteredDataset, groups []string) (string, error) {
	plan, training, err := conversationSplit(registered.Dataset, groups)
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

// conversationGroups names the group each record belongs to: the digest of how its conversation
// opens, the text before its first reply when it learns replies (the system prompt and the
// opening turns), or its first window when it learns every token, so the variants of one chat (a
// prefix and its continuation, conversations that differ only in a reply) share a group.
func conversationGroups(records [][]byte) ([]string, error) {
	groups := make([]string, len(records))
	for index, data := range records {
		var record conversationRecord
		if err := json.Unmarshal(data, &record); err != nil || len(record.Segments) == 0 {
			return nil, fmt.Errorf("training workflow: conversation record %d is unreadable: %w", index, errors.Join(err, errors.New("no segments")))
		}
		opening := record.Segments[0].Text
		if record.Segments[0].Trained {
			opening = opening[:min(len(opening), conversationWindow)]
		}
		sum := sha256.Sum256([]byte(opening))
		groups[index] = hex.EncodeToString(sum[:])
	}
	return groups, nil
}

// conversationSplit divides the dataset's records into a training and a held-out membership,
// by group: the held-out groups are the first by the digest of their first record, taken whole
// until a tenth of the records, at most the cap, are held out, so a dataset always splits the same way,
// no conversation's variants sit on both sides of the verdict, and neither side is ever empty.
func conversationSplit(source artifact.ID, groups []string) (dataset.SplitPlan, dataset.Membership, error) {
	count := len(groups)
	members := map[string][]dataset.Record{}
	for index, group := range groups {
		id := dataset.RecordID(source, dataset.InventoryAssetName, uint64(index))
		members[group] = append(members[group], dataset.Record{ID: id, Group: group})
	}
	byDigest := func(left, right dataset.Record) int {
		leftDigest, rightDigest := sha256.Sum256([]byte(left.ID)), sha256.Sum256([]byte(right.ID))
		return bytes.Compare(leftDigest[:], rightDigest[:])
	}
	// A group sorts by its first record's digest, so conversations that all open differently
	// are held out in the order their records' digests give, as they were before groups.
	for _, records := range members {
		slices.SortFunc(records, byDigest)
	}
	order := slices.Collect(maps.Keys(members))
	slices.SortFunc(order, func(left, right string) int { return byDigest(members[left][0], members[right][0]) })
	target := min(max(count/conversationHeldoutShare, 1), conversationHeldoutRecords)
	var held, kept []dataset.Record
	for _, group := range order {
		// A group joins the held-out side while the side is short of its share and the training side keeps a record.
		if len(held) < target && count-len(held)-len(members[group]) > 0 {
			held = append(held, members[group]...)
		} else {
			kept = append(kept, members[group]...)
		}
	}
	if len(held) == 0 || len(kept) == 0 {
		return dataset.SplitPlan{}, dataset.Membership{}, fmt.Errorf("training workflow: the conversations open in %d distinct way(s), too few to hold one out to judge the rest", len(order))
	}
	slices.SortFunc(held, byDigest)
	slices.SortFunc(kept, byDigest)
	// Partitions list by name, the order a split canonicalizes them in.
	plan, err := dataset.SplitOf(source, 0, []dataset.PartitionRecords{
		{Name: conversationHeldout, Records: held}, {Name: conversationTrain, Records: kept},
	})
	if err != nil {
		return dataset.SplitPlan{}, dataset.Membership{}, err
	}
	return plan, plan.Memberships[1], nil // the training partition, the second by name
}
