package main

import (
	"encoding/json"
	"testing"
)

// TestAUserSizeRequestLeavesItsSizeToTheModel: a verified request loses the size it states, so the model's own defaults
// fill it, and keeps what it asks for. One that states no size has none to lift.
func TestAUserSizeRequestLeavesItsSizeToTheModel(t *testing.T) {
	t.Parallel()
	sized, stated, err := userSizeRequest(`{"prompt":"a fox","frames":21,"width":32,"height":16,"steps":2,"seed":7}`)
	if err != nil || !stated {
		t.Fatalf("stated %t (%v)", stated, err)
	}
	var fields map[string]any
	if err := json.Unmarshal([]byte(sized), &fields); err != nil {
		t.Fatal(err)
	}
	if len(fields) != 2 || fields["prompt"] != "a fox" || fields["seed"] == nil {
		t.Fatalf("the request reads %s, want its prompt and seed alone", sized)
	}
	if _, stated, err := userSizeRequest(`{"prompt":"a fox"}`); err != nil || stated {
		t.Fatalf("a request of no size: stated %t (%v)", stated, err)
	}
	if _, _, err := userSizeRequest("not a request"); err == nil {
		t.Fatal("a request that is no JSON reads as one")
	}
}
