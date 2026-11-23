//go:build windows

package mediacapability

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAnUndeclaredPresetIsRefusedAtAnySize holds served-answers/preset (SQA-425): a MOSS request naming a preset its
// model does not declare, and a picture the store does not hold, is refused for its preset whether it states its size
// or not, before the session is reached and before the picture is read. Request.Sized had read the preset only for a
// request stating no size, so with 1024 by 576 stated the request went on and was refused for its missing picture.
func TestAnUndeclaredPresetIsRefusedAtAnySize(t *testing.T) {
	t.Parallel()
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	store := overgodbtest.Open(t, t.TempDir())
	absent := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("never stored")))(t)
	reached := 0
	session := func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		reached++
		return "drawn", nil
	}
	serve := capabilityruntime.Reading(preparedFirst(session), capabilityruntime.Contents(nil, pictureFiles))
	for _, request := range []string{
		`{"prompt":"a fox","steps":8,"preset":"no-such-preset","image":"` + absent.String() + `"}`,
		`{"prompt":"a fox","steps":8,"width":1024,"height":576,"preset":"no-such-preset","image":"` + absent.String() + `"}`,
	} {
		_, err := serve(t.Context(), store, root, modelrecipe.CapabilityEvidenceSelection{}, request)
		if err == nil || !strings.Contains(err.Error(), `preset "no-such-preset"`) || reached != 0 {
			t.Errorf("%s was refused with %v, the session reached %d times; want its preset refused before the session", request, err, reached)
		}
	}
}
