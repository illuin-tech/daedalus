declare module "plotly.js-basic-dist-min" {
  type Traces = readonly Record<string, unknown>[];
  const Plotly: {
    newPlot(target: HTMLElement, traces: Traces, layout: Record<string, unknown>, config: Record<string, unknown>): Promise<unknown>;
    react(target: HTMLElement, traces: Traces, layout: Record<string, unknown>, config: Record<string, unknown>): Promise<unknown>;
    animate(target: HTMLElement, frame: Record<string, unknown>, options: Record<string, unknown>): Promise<unknown>;
    restyle(target: HTMLElement, update: Record<string, unknown>, traces?: number[]): Promise<unknown>;
    relayout(target: HTMLElement, update: Record<string, unknown>): Promise<unknown>;
    Plots: {
      resize(target: HTMLElement): Promise<unknown>;
    };
  };

  export default Plotly;
}
