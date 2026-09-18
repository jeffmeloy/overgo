package plan

import (
	"strings"
	"testing"
	"time"
)

func TestAcceptedProgressUsesCompletionAuthority(t *testing.T) {
	for _, name := range []string{"prerequisite", "sibling", "transitive", "unrelated", "foreign owner"} {
		t.Run(name, func(t *testing.T) {
			document := standardCompletionPlan()
			document.Items[0].Owner = "colibri"
			document.Items[1].Owner = "colibri"
			item, step := "dependent", "do"
			switch name {
			case "sibling":
				document.Items[0].Steps = append(document.Items[0].Steps, Step{ID: "remaining", Status: StatusOpen, Verify: "go test ./..."})
				item, step = "root", "remaining"
			case "transitive":
				document.Items[1].Steps[0].DependsOn = []string{"middle/do"}
				document.Items = append(document.Items, Item{ID: "middle", Owner: "colibri", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"root/do"}}}})
			case "unrelated":
				document.Items[1].Steps[0].DependsOn = nil
			case "foreign owner":
				document.Items[0].Owner = "other"
			}
			fixture := newCompletionFixture(t, document, "root", "do")
			before := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
			fixture.commit(fixture.canonicalMessage(), true)
			authority, err := resolveFixture(fixture, fixture.child, "HEAD")
			if err != nil {
				t.Fatal(err)
			}
			accepted, err := authority.AcceptedProgressSince(t.Context(), fixture.child, item, step, before)
			want := name != "unrelated" && name != "foreign owner"
			if err != nil || accepted != want {
				t.Fatalf("accepted=%v want=%v err=%v", accepted, want, err)
			}
			if accepted, err := authority.AcceptedProgressSince(t.Context(), fixture.child, item, step, fixture.completionHash); err != nil || accepted {
				t.Fatalf("unchanged receipt earned progress: %v %v", accepted, err)
			}
			if accepted, err := authority.AcceptedProgressSince(t.Context(), fixture.child, item, step, strings.Repeat("0", 40)); err == nil || accepted {
				t.Fatal("foreign checkpoint accepted")
			}
		})
	}
}

func TestAcceptedProgressRejectsReplanAndHeadMovement(t *testing.T) {
	document := standardCompletionPlan()
	document.Items[1].Steps[0].DependsOn = nil
	fixture := newCompletionFixture(t, document, "root", "do")
	fixture.commit(fixture.canonicalMessage(), true)
	checkpoint := fixture.completionHash
	// The old accepted receipt becomes relevant only after replanning. It is
	// still old work, even with a new non-completion commit at HEAD.
	fixture.child.Items[0].Steps[0].DependsOn = []string{"root/do"}
	if err := Save(fixture.repository+"/"+Path, fixture.child); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	runGit(t, fixture.repository, nil, "commit", "-q", "-m", "replan only")
	authority, err := resolveFixture(fixture, fixture.child, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if accepted, err := authority.AcceptedProgressSince(t.Context(), fixture.child, "dependent", "do", checkpoint); err != nil || accepted {
		t.Fatalf("replan earned progress: %v %v", accepted, err)
	}
}

func TestAcceptedProgressRejectsMergeCompletion(t *testing.T) {
	document := standardCompletionPlan()
	document.Items = append(document.Items, Item{ID: "seed", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}})
	fixture := newCompletionFixture(t, document, "seed", "do")
	fixture.commit(fixture.canonicalMessage(), true)
	fixture.parent, fixture.preAdvance = fixture.child, fixture.child
	var err error
	fixture.child, err = pruneCompletionFixture(fixture.parent, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	fixture.item, fixture.step = "root", "do"
	fixture.rotatePreparation(strings.Repeat("b", 64), time.Unix(2, 0))
	checkpoint := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	runGit(t, fixture.repository, nil, "checkout", "-qb", "progress-side")
	runGit(t, fixture.repository, nil, "commit", "-q", "--allow-empty", "-m", "incoming work")
	runGit(t, fixture.repository, nil, "checkout", "-q", "--detach", checkpoint)
	runGit(t, fixture.repository, nil, "merge", "--no-ff", "--no-commit", "progress-side")
	fixture.commit(fixture.canonicalMessage(), true)
	authority, err := resolveFixture(fixture, fixture.child, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if accepted, err := authority.AcceptedProgressSince(t.Context(), fixture.child, "dependent", "do", checkpoint); err != nil || accepted {
		t.Fatalf("accepted merge earned implementation progress: %v %v", accepted, err)
	}
}
