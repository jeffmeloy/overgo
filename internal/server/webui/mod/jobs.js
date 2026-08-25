(function () {
  "use strict";
  window.overgo.workflowWorkspace({
    id: "training-jobs", scope: "training",
    renderEvidence: window.overgo.trainingEvidence.renderOperation,
  });
  window.overgo.workflowWorkspace({ id: "export-jobs", scope: "export" });
})();
