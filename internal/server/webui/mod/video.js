/* Video: generation and reference-guided editing as composer modes over their own threads. */
window.overgo.generationTab("video-gen", "video-gen", { placeholder: "describe the video to generate", sendLabel: "generate" });
window.overgo.generationTab("video-edit", "video-edit", {
  placeholder: "describe the edit to apply", sendLabel: "edit", kinds: ["video"], multiple: false, attachLabel: "source video",
});
