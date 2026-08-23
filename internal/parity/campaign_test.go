package parity

import (
	"errors"
	"math"
	"path/filepath"
	"testing"
)

func TestCampaignStopsAfterFailure(t *testing.T) {
	campaign := NewCampaign(filepath.Join(t.TempDir(), "parity.log"))
	campaign.Run("pass", func() (Verdict, float64, string, error) { return Pass, math.NaN(), "ok", nil })
	campaign.Run("fail", func() (Verdict, float64, string, error) { return Pass, 0, "", errors.New("failed") })
	campaign.Run("unreached", func() (Verdict, float64, string, error) { return Pass, 0, "", nil })
	if !campaign.Failed() || campaign.Total() != 2 || campaign.Count(Pass) != 1 || campaign.Count(Failed) != 1 {
		t.Fatalf("failed=%t total=%d pass=%d fail=%d", campaign.Failed(), campaign.Total(), campaign.Count(Pass), campaign.Count(Failed))
	}
}
