package audioparity

import (
	"context"
	"errors"
	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
)

// requireActivity checks canonical activity output in parity fixtures.
func requireActivity(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipecontract.ActivitySegments, error) {
	activityContract := artifact.JSONContract(artifact.KindOutput, "overgo/audio-activity/v1")
	var result recipecontract.ActivitySegments
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil || !found {
		return result, errors.Join(errors.New("speech activity: output absent"), err)
	}
	if err := activityContract.ValidateContent(content, id); err != nil {
		return result, err
	}
	if err := strictjson.DecodeBytes(content.Data, &result); err != nil {
		return result, err
	}
	if err := result.Validate(); err != nil {
		return recipecontract.ActivitySegments{}, err
	}
	canonical, err := artifact.JSONContent(activityContract, result)
	if err != nil || canonical.Descriptor.ID != id {
		return recipecontract.ActivitySegments{}, errors.Join(errors.New("speech activity: output is not canonical"), err)
	}
	return result, nil
}
