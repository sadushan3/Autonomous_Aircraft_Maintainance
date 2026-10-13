import { createBrowserRouter } from "react-router";
import { Root } from "./components/Root";
import { Overview } from "./pages/Overview";
import { Upload } from "./pages/Upload";
import { HealthMonitoring } from "./pages/HealthMonitoring";
import { AnomalyDetection } from "./pages/AnomalyDetection";
import { FailureRisk } from "./pages/FailureRisk";
import { RULPrediction } from "./pages/RULPrediction";
import { DigitalTwin } from "./pages/DigitalTwin";
import { AIReasoning } from "./pages/AIReasoning";
import { MaintenanceSupervisor } from "./pages/MaintenanceSupervisor";
import { Reports } from "./pages/Reports";
import { Settings } from "./pages/Settings";
import { Confidence } from "./pages/Confidence";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: Overview },
      { path: "upload", Component: Upload },
      { path: "health", Component: HealthMonitoring },
      { path: "anomaly", Component: AnomalyDetection },
      { path: "failure-risk", Component: FailureRisk },
      { path: "rul", Component: RULPrediction },
      { path: "digital-twin", Component: DigitalTwin },
      { path: "ai-reasoning", Component: AIReasoning },
      { path: "confidence", Component: Confidence },
      { path: "maintenance", Component: MaintenanceSupervisor },
      { path: "reports", Component: Reports },
      { path: "settings", Component: Settings },
    ],
  },
]);
