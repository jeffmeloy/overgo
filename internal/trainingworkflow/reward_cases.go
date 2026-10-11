package trainingworkflow

import (
	"cmp"
	"context"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

// producedCase is one prompt a reward session samples answers to, and where
// the session's stream stands once it is consumed.
type producedCase struct {
	name  string
	state trainingdata.StreamState
}

// rewardCasesResume reads the prompts of a reward session that samples its
// own answers, and orders the ones this session consumes. The dataset is the
// suite of prompts, one record a prompt, and the stream's position is the
// cursor over them, which a checkpoint keeps and a resumed session continues
// from.
func rewardCasesResume(ctx context.Context, raw []byte, steps int, resume *trainingdata.StreamState) (evaluation.InstructionPrompts, []producedCase, trainingdata.StreamState, batchAuthority, error) {
	failed := func(err error) (evaluation.InstructionPrompts, []producedCase, trainingdata.StreamState, batchAuthority, error) {
		return evaluation.InstructionPrompts{}, nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	prompts, err := evaluation.ParseInstructionPrompts(raw)
	if err != nil {
		return failed(err)
	}
	authority, processor, err := datasetAuthority(raw, "reward-prompts")
	if err != nil {
		return failed(err)
	}
	names := prompts.Names()
	records := make([]trainingdata.RawRecord, len(names))
	for index, name := range names {
		records[index] = trainingdata.RawRecord{ID: name, Group: name, Data: []byte(name)}
	}
	materialized, err := trainingdata.MaterializeRecords(trainingdata.Authority{
		Dataset: authority.Dataset, Split: authority.Split, Processors: []artifact.ID{processor}, Signature: textSignature,
	}, processor, records, trainingdata.ProcessorBinding{
		Artifact: processor, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "reward-prompt-name"),
	})
	if err != nil {
		return failed(err)
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, resume)
	if err != nil {
		return failed(err)
	}
	batcher, err := trainingdata.NewBatcher(stream, unitBatchPolicy)
	if err != nil {
		return failed(err)
	}
	cases := make([]producedCase, cmp.Or(steps, len(names)))
	for step := range cases {
		batch, err := batcher.Next(ctx)
		if err != nil {
			return failed(err)
		}
		cases[step] = producedCase{name: string(batch.Examples[0].Values[0].Data), state: batch.State}
	}
	return prompts, cases, stream.Snapshot(), authority, nil
}
