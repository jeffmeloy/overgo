package media

import "testing"

func TestPatchGrid(t *testing.T) {
	height, width, err := PatchGrid(16, 8, 4)
	if err != nil {
		t.Fatal(err)
	}
	if height != 4 || width != 2 {
		t.Fatalf("grid=%dx%d", height, width)
	}
	if _, _, err := PatchGrid(15, 8, 4); err == nil {
		t.Fatal("accepted inexact patch grid")
	}
	if _, _, err := PatchGrid(16, 8, 0); err == nil {
		t.Fatal("accepted zero patch")
	}
}

func TestPatchVectorWidth(t *testing.T) {
	width, err := PatchVectorWidth(16, 2)
	if err != nil || width != 64 {
		t.Fatalf("width=%d err=%v", width, err)
	}
	if _, err := PatchVectorWidth(16, 0); err == nil {
		t.Fatal("accepted zero patch")
	}
}
