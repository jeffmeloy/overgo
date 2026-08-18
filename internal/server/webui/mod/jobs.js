(function () {
  "use strict";
  window.overgo.workflowWorkspace({
    id: "training-jobs", label: "Train", section: "training", scope: "training",
    renderEvidence: window.overgo.trainingEvidence.renderOperation,
  });
  window.overgo.workflowWorkspace({ id: "export-jobs", label: "Export", section: "training", scope: "export" });
})();
